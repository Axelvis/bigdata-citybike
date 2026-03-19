import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, sum as spark_sum, avg as spark_avg, month, dayofweek, desc, year
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

if __name__ == '__main__':
    print("🚀 Démarrage de la session PySpark Avancée...")
    
    dossier_temp = os.path.abspath("./spark_temp")
    os.makedirs(dossier_temp, exist_ok=True)

    # Configuration mémoire conservée (4g) pour éviter de saturer ta machine
    spark = SparkSession.builder \
        .appName("CitiBike_Reequilibrage_ML") \
        .config("spark.sql.shuffle.partitions", "50") \
        .config("spark.driver.memory", "4g") \
        .config("spark.local.dir", dossier_temp) \
        .getOrCreate()

    # ---------------------------------------------------------
    # 1. PRÉPARATION DU FLUX NET (Ingénierie des données)
    # ---------------------------------------------------------
    print("📂 Lecture de la base de données unifiée...")
    df_final = spark.read.parquet("data/dataset_horaire_final")
    
    col_start = "start_station_name" if "start_station_name" in df_final.columns else "start station name"
    col_end = "end_station_name" if "end_station_name" in df_final.columns else "end station name"

    print("🔍 Identification des 50 stations principales...")
    top_stations_rows = df_final.groupBy(col_start).count().orderBy(desc("count")).limit(50).collect()
    top_stations = [row[col_start] for row in top_stations_rows if row[col_start] is not None]

    print("🔄 Calcul des flux (Départs vs Arrivées) sur le Top 50...")
    
    df_departs = df_final.select(
        col(col_start).alias("station"), 
        col("date_trajet"), col("heure_trajet"), 
        col("temperature_c"), col("precipitations_mm"), col("vent_kmh"),
        lit(-1).alias("mouvement")
    ).filter(col("station").isin(top_stations))

    df_arrivees = df_final.select(
        col(col_end).alias("station"), 
        col("date_trajet"), col("heure_trajet"), 
        col("temperature_c"), col("precipitations_mm"), col("vent_kmh"),
        lit(1).alias("mouvement")
    ).filter(col("station").isin(top_stations))

    df_mouvements = df_departs.unionAll(df_arrivees)

    # ---------------------------------------------------------
    # 2. AGRÉGATION
    # ---------------------------------------------------------
    print("⚡ Agrégation par station et par heure...")
    
    df_mouvements = df_mouvements.withColumn("annee", year("date_trajet")) \
                                 .withColumn("mois", month("date_trajet")) \
                                 .withColumn("jour_semaine", dayofweek("date_trajet"))

    df_ml = df_mouvements.groupBy(
        "station", "annee", "jour_semaine", "mois", "heure_trajet", 
        "temperature_c", "precipitations_mm", "vent_kmh"
    ).agg(spark_sum("mouvement").alias("flux_net"))

    print("💾 Mise en cache des données...")
    df_ml.cache()
    df_ml.count() 

    # ---------------------------------------------------------
    # 3. PIPELINE ML ET SÉPARATION CHRONOLOGIQUE STRICTE
    # ---------------------------------------------------------
    print("🤖 Construction du Pipeline SparkML...")

    indexer = StringIndexer(inputCol="station", outputCol="station_index", handleInvalid="keep")

    assembleur = VectorAssembler(
        inputCols=["station_index", "jour_semaine", "mois", "heure_trajet", 
                   "temperature_c", "precipitations_mm", "vent_kmh"],
        outputCol="features"
    )

    # Allègement du modèle pour éviter l'OOM (Out Of Memory)
    rf = RandomForestRegressor(featuresCol="features", labelCol="flux_net", numTrees=30, maxDepth=7, maxBins=64)
    pipeline = Pipeline(stages=[indexer, assembleur, rf])

    # SÉPARATION STRICTE : On ne garde que 2023-2024 pour l'entraînement (Post-COVID, plus pertinent et plus léger)
    print("✂️ Séparation Train/Test (Train=2023-2024 | Test=2025)...")
    train_data = df_ml.filter((col("annee") >= 2023) & (col("annee") < 2025))
    test_data  = df_ml.filter(col("annee") >= 2025)

    print("⏳ Entraînement du modèle (Mémoire optimisée)...")
    modele = pipeline.fit(train_data)

    # ---------------------------------------------------------
    # 4. ÉVALUATION ET SAUVEGARDE
    # ---------------------------------------------------------
    print("🎯 Évaluation sur les données de TEST (Totalement inconnues de l'IA)...")
    predictions = modele.transform(test_data)

    evaluator_r2 = RegressionEvaluator(labelCol="flux_net", predictionCol="prediction", metricName="r2")
    evaluator_rmse = RegressionEvaluator(labelCol="flux_net", predictionCol="prediction", metricName="rmse")
    
    print("-" * 50)
    print(f"R² (Explication de la variance) : {evaluator_r2.evaluate(predictions):.2f}")
    print(f"RMSE (Erreur moyenne)           : +/- {evaluator_rmse.evaluate(predictions):.1f} vélos/heure")
    print("-" * 50)

    chemin_modele = "data/modele_sparkml_reequilibrage"
    modele.write().overwrite().save(chemin_modele)
    print(f"✅ Modèle sauvegardé dans : {chemin_modele}")

    # ---------------------------------------------------------
    # 5. VISUALISATION OPÉRATIONNELLE : SEMAINE TYPE (Top Station)
    # ---------------------------------------------------------
    print("📈 Génération de la courbe (Semaine type en Janvier 2025 pour la Top Station)...")
    
    # 1. Identifier la plus grosse station
    top_1_station = top_stations[0]
    print(f"📍 Station analysée : {top_1_station}")

    # 2. Filtrer le Test Set (2025) UNIQUEMENT pour cette station, pour Janvier,
    # et moyenner par jour de la semaine et heure pour créer une "Semaine Type".
    df_plot_spark = predictions.filter(
        (col("station") == top_1_station) & 
        (col("mois") == 1)
    ).groupBy("jour_semaine", "heure_trajet").agg(
        spark_avg("flux_net").alias("flux_net"),
        spark_avg("prediction").alias("prediction")
    ).orderBy("jour_semaine", "heure_trajet")
    
    # 3. Rapatriement sur Pandas
    pdf_plot = df_plot_spark.toPandas()

    # 4. Création d'un axe X continu (Heure 0 à Heure 167 d'une semaine)
    # dayofweek dans Spark : 1 = Dimanche. On fait -1 pour démarrer à 0.
    pdf_plot['heure_absolue'] = (pdf_plot['jour_semaine'] - 1) * 24 + pdf_plot['heure_trajet']

    # 5. Dessin du graphique
    plt.figure(figsize=(16, 6))
    sns.set_theme(style="whitegrid")
    
    sns.lineplot(data=pdf_plot, x="heure_absolue", y="flux_net", label="Vraie donnée (Réalité Moyenne)", color="#2ecc71", linewidth=2)
    sns.lineplot(data=pdf_plot, x="heure_absolue", y="prediction", label="Prédiction IA", color="#e74c3c", linewidth=2, linestyle="--")
    
    # Ligne zéro pour bien distinguer la zone "La station se remplit" vs "La station se vide"
    plt.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)

    # Remplacement des numéros d'heures par le nom des jours
    jours_noms = ["Dimanche", "Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi"]
    plt.xticks(ticks=[i*24 + 12 for i in range(7)], labels=jours_noms, fontsize=12)

    plt.title(f"Anticipation du Flux Net - Semaine Type en Janvier 2025\n(Station : {top_1_station})", fontsize=16, fontweight='bold')
    plt.xlabel("Jour de la semaine", fontsize=12)
    plt.ylabel("Flux net (Arrivées - Départs)", fontsize=12)
    plt.legend(fontsize=12, loc="upper right")
    
    fichier_courbe_focus = "data/ml_predictions_top1_station_semaine.png"
    plt.savefig(fichier_courbe_focus, bbox_inches="tight")
    print(f"🖼️ Courbe opérationnelle sauvegardée sous : {fichier_courbe_focus}")

    df_ml.unpersist()
    spark.stop()
