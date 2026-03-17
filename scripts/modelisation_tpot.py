import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import count, month, dayofweek
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ---------------------------------------------------------
# SÉCURITÉ MULTIPROCESSING (Le garde du corps)
# Tout le code doit être indenté sous cette ligne !
# ---------------------------------------------------------
if __name__ == '__main__':

    print("🚀 Démarrage de la session PySpark pour l'agrégation...")
    spark = SparkSession.builder \
        .appName("CitiBike_Aggregation") \
        .getOrCreate()

    # 1. AGRÉGATION AVEC PYSPARK
    print("📊 Lecture de la base de données unifiée...")
    df_final = spark.read.parquet("data/dataset_final_modelisation")

    df_features = df_final.withColumn("mois", month("date_trajet")) \
                          .withColumn("jour_semaine", dayofweek("date_trajet"))

    print("🔄 Calcul du nombre de trajets quotidiens...")
    df_quotidien = df_features.groupBy(
        "date_trajet", "temperature_f", "precipitations_pouces", "mois", "jour_semaine"
    ).agg(count("*").alias("total_locations"))

    # 2. BASCULE VERS PANDAS
    print("📉 Conversion en Pandas pour TPOT...")
    df_ml = df_quotidien.toPandas()
    df_ml = df_ml.dropna()

    X = df_ml[["temperature_f", "precipitations_pouces", "mois", "jour_semaine"]]
    y = df_ml["total_locations"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. AUTO-MACHINE LEARNING AVEC TPOT
    print("\n🤖 Lancement de TPOT (Recherche du meilleur modèle)...")
    print("⚠️ Cela peut prendre plusieurs minutes. Laissez l'IA travailler !")

    tpot = TPOTRegressor(
        generations=5, 
        population_size=20, 
        verbose=2, 
        random_state=42, 
        n_jobs=1  # On garde 1 pour éviter de surcharger la RAM de la VM
    )

    tpot.fit(X_train, y_train)

    # 4. RÉSULTATS ET EXPORTATION
    print("\n✅ Entraînement terminé !")

    predictions = tpot.predict(X_test)
    score_r2 = r2_score(y_test, predictions)

    print("-" * 50)
    print(f"🎯 Score R² du meilleur modèle : {score_r2:.2f}")
    print("-" * 50)

    fichier_export = "src/meilleur_modele_pipeline.py"
    tpot.export(fichier_export)
    print(f"💾 Le code source du meilleur pipeline a été sauvegardé dans : {fichier_export}")
