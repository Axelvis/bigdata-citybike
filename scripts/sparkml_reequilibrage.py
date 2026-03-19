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
    
    # Création d'un dossier temporaire local pour éviter de saturer le /tmp racine
    dossier_temp = os.path.abspath("./spark_temp")
    os.makedirs(dossier_temp, exist_ok=True)

    # Configuration optimisée
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

    # OPTIMISATION : On isole les 50 plus grandes stations AVANT les gros calculs
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
    
    # On extrait l'année pour pouvoir séparer le dataset plus tard
    df_mouvements = df_mouvements.withColumn("annee", year("date_trajet")) \
                                 .withColumn("mois", month("date_trajet")) \
                                 .withColumn("jour_semaine", dayofweek("date_trajet"))

    # Calcul du flux net par heure
    df_ml = df_mouvements.groupBy(
        "station", "annee", "jour_semaine", "mois", "heure_trajet", 
        "temperature_c", "precipitations_mm", "vent_kmh"
    ).agg(spark_sum("mouvement").alias("flux_net"))

    print("💾 Mise en cache des données...")
    df_ml.cache()
    df_ml.count() # Force l'exécution immédiate

    # ---------------------------------------------------------
    # 3. PIPELINE ML ET SÉPARATION CHRONOLOGIQUE STRICTE
    # ---------------------------------------------------------
    print("🤖 Construction du Pipeline SparkML...")

    # handleInvalid="keep" permet d'éviter un crash si le dataset de test a une donnée bizarre
    indexer = StringIndexer(inputCol="station", outputCol="station_index", handleInvalid="keep")

    assembleur = VectorAssembler(
        inputCols=["station_index", "jour_semaine", "mois", "heure_trajet", 
                   "temperature_c", "precipitations_mm", "vent_kmh"],
        outputCol="features"
    )

    rf = RandomForestRegressor(featuresCol="features", labelCol="flux_net", numTrees=50, maxDepth=10, maxBins=64)
    pipeline = Pipeline(stages=[indexer, assembleur, rf])

    # SÉPARATION STRICTE DANS LE TEMPS
    print("✂️ Séparation Train/Test (Test = Années >= 2025)...")
    train_data = df_ml.filter(col("annee") < 2025)
    test_data  = df_ml.filter(col("annee") >= 2025)

    print("⏳ Entraînement du modèle sur l'historique (cela peut prendre quelques minutes)...")
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
    # 5. VISUALISATION STRICTE : RÉALITÉ vs PRÉDICTION (TEST SET)
    # ---------------------------------------------------------
    print("📈 Génération de la courbe (Réalité vs Prédiction sur le Test Set)...")
    
    # L'agrégation pour le graphique ne se fait QUE sur la table `predictions` (qui est le test set)
    df_plot_spark = predictions.groupBy("heure_trajet").agg(
        spark_avg("flux_net").alias("Vrai_Flux"),
        spark_avg("prediction").alias("Flux_Predit")
    ).orderBy("heure_trajet")
    
    pdf_plot = df_plot_spark.toPandas()

    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    
    sns.lineplot(data=pdf_plot, x="heure_trajet", y="Vrai_Flux", label="Vraie donnée (Test Set)", color="#2ecc71", linewidth=2.5, marker="o")
    sns.lineplot(data=pdf_plot, x="heure_trajet", y="Flux_Predit", label="Prédiction Modèle", color="#e74c3c", linewidth=2.5, linestyle="--", marker="X")
    
    plt.title("Comparaison Stricte : Flux Net Réel vs Prédiction par Heure (Test Set)", fontsize=16, fontweight='bold')
    plt.xlabel("Heure de la journée (0-23h)", fontsize=12)
    plt.ylabel("Flux net moyen (Arrivées - Départs)", fontsize=12)
    plt.xticks(range(0, 24))
    plt.legend(fontsize=12)
    
    fichier_courbe = "data/ml_predictions_vs_realite.png"
    plt.savefig(fichier_courbe, bbox_inches="tight")
    print(f"🖼️ Courbe sauvegardée sous : {fichier_courbe}")

    df_ml.unpersist()
    spark.stop()
