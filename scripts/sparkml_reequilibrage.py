import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, sum as spark_sum, month, dayofweek
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

if __name__ == '__main__':
    print("🚀 Démarrage de la session PySpark Avancée...")
    # OPTIMISATION 1 : Configuration de la mémoire et des partitions Spark
    spark = SparkSession.builder \
        .appName("CitiBike_Reequilibrage_ML") \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.driver.memory", "8g") \
        .getOrCreate()

    # ---------------------------------------------------------
    # 1. PRÉPARATION DU FLUX NET (Ingénierie des données)
    # ---------------------------------------------------------
    print("📂 Lecture de la base de données unifiée...")
    df_final = spark.read.parquet("data/dataset_horaire_final")
    
    # Identification sécurisée des colonnes de stations
    col_start = "start_station_name" if "start_station_name" in df_final.columns else "start station name"
    col_end = "end_station_name" if "end_station_name" in df_final.columns else "end station name"

    print("🔄 Calcul des flux (Départs vs Arrivées)...")
    
    # ASTUCE BIG DATA : Au lieu d'une jointure lourde, on crée des événements +1 et -1
    # 1. Les départs vident la station (-1)
    df_departs = df_final.select(
        col(col_start).alias("station"), 
        col("date_trajet"), col("heure_trajet"), 
        col("temperature_c"), col("precipitations_mm"), col("vent_kmh"),
        lit(-1).alias("mouvement")
    ).dropna(subset=["station"])

    # 2. Les arrivées remplissent la station (+1)
    df_arrivees = df_final.select(
        col(col_end).alias("station"), 
        col("date_trajet"), col("heure_trajet"), 
        col("temperature_c"), col("precipitations_mm"), col("vent_kmh"),
        lit(1).alias("mouvement")
    ).dropna(subset=["station"])

    # On fusionne les deux tableaux
    df_mouvements = df_departs.unionAll(df_arrivees)

    # ---------------------------------------------------------
    # 2. AGRÉGATION ET OPTIMISATION (Partitions & Cache)
    # ---------------------------------------------------------
    print("⚡ Agrégation par station et par heure...")
    
    # Extraction du calendrier
    df_mouvements = df_mouvements.withColumn("mois", month("date_trajet")) \
                                 .withColumn("jour_semaine", dayofweek("date_trajet"))

    # OPTIMISATION 2 : Repartitionnement intelligent par station pour accélérer le GroupBy
    df_mouvements = df_mouvements.repartition("station")

    # Calcul du flux net par station, par jour et par heure
    df_ml_brut = df_mouvements.groupBy(
        "station", "jour_semaine", "mois", "heure_trajet", 
        "temperature_c", "precipitations_mm", "vent_kmh"
    ).agg(spark_sum("mouvement").alias("flux_net"))

    # On se limite aux 50 plus grandes stations pour un modèle très précis et rapide
    top_stations = df_ml_brut.groupBy("station").count().orderBy(col("count").desc()).limit(50)
    df_ml = df_ml_brut.join(top_stations, "station", "left_semi")

    # OPTIMISATION 3 : Mise en cache !
    # Le dataframe va être lu plusieurs fois par l'algorithme ML. 
    # Le garder en RAM évite de recalculer tout ce qui précède à chaque itération.
    print("💾 Mise en cache des données d'entraînement...")
    df_ml.cache()
    # Action factice pour forcer le chargement immédiat dans le cache
    df_ml.count() 

    # ---------------------------------------------------------
    # 3. PIPELINE MACHINE LEARNING (SparkML)
    # ---------------------------------------------------------
    print("🤖 Construction du Pipeline SparkML...")

    # A. Convertir les noms de stations (Texte) en indices numériques (0, 1, 2...)
    indexer = StringIndexer(inputCol="station", outputCol="station_index")

    # B. Assembler toutes les variables prédictives dans un seul vecteur "features"
    assembleur = VectorAssembler(
        inputCols=["station_index", "jour_semaine", "mois", "heure_trajet", 
                   "temperature_c", "precipitations_mm", "vent_kmh"],
        outputCol="features"
    )

    # C. L'algorithme d'IA (Random Forest est excellent pour ce type de données)
    rf = RandomForestRegressor(featuresCol="features", labelCol="flux_net", numTrees=50, maxDepth=10, maxBins=64)
    # On assemble le pipeline complet
    pipeline = Pipeline(stages=[indexer, assembleur, rf])

    # Séparation Train / Test
    train_data, test_data = df_ml.randomSplit([0.8, 0.2], seed=42)

    print("⏳ Entraînement du modèle distribué (cela peut prendre quelques minutes)...")
    modele = pipeline.fit(train_data)

    # ---------------------------------------------------------
    # 4. ÉVALUATION ET SAUVEGARDE
    # ---------------------------------------------------------
    print("🎯 Évaluation des performances...")
    predictions = modele.transform(test_data)

    evaluator_r2 = RegressionEvaluator(labelCol="flux_net", predictionCol="prediction", metricName="r2")
    evaluator_rmse = RegressionEvaluator(labelCol="flux_net", predictionCol="prediction", metricName="rmse")
    
    r2 = evaluator_r2.evaluate(predictions)
    rmse = evaluator_rmse.evaluate(predictions)

    print("-" * 50)
    print(f"R² (Précision globale) : {r2:.2f}")
    print(f"RMSE (Erreur moyenne)  : +/- {rmse:.1f} vélos par heure")
    print("-" * 50)

    # Sauvegarde du pipeline ML complet au format natif Spark
    chemin_modele = "data/modele_sparkml_reequilibrage"
    modele.write().overwrite().save(chemin_modele)
    
    print(f"✅ Modèle distribué sauvegardé dans : {chemin_modele}")
    print("Prêt à prévoir les camions de rééquilibrage de demain !")

    # Libération de la RAM
    df_ml.unpersist()
    spark.stop()
