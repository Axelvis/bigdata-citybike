import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import count, month, dayofweek
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

print("🚀 Démarrage de la session PySpark pour l'agrégation...")
spark = SparkSession.builder \
    .appName("CitiBike_Aggregation") \
    .getOrCreate()

# ---------------------------------------------------------
# 1. AGRÉGATION AVEC PYSPARK (Big Data)
# ---------------------------------------------------------
print("📊 Lecture de la base de données unifiée...")
df_final = spark.read.parquet("data/dataset_final_modelisation")

# On extrait le mois et le jour de la semaine pour donner plus de contexte à l'IA
df_features = df_final.withColumn("mois", month("date_trajet")) \
                      .withColumn("jour_semaine", dayofweek("date_trajet"))

# On groupe par jour pour compter le nombre total de locations
print("🔄 Calcul du nombre de trajets quotidiens...")
df_quotidien = df_features.groupBy(
    "date_trajet", "temperature_f", "precipitations_pouces", "mois", "jour_semaine"
).agg(count("*").alias("total_locations"))

# ---------------------------------------------------------
# 2. BASCULE VERS PANDAS (Machine Learning)
# ---------------------------------------------------------
print("📉 Conversion en Pandas pour TPOT...")
df_ml = df_quotidien.toPandas()

# Nettoyage final : suppression des éventuelles lignes où la météo serait vide (NaN)
df_ml = df_ml.dropna()

# Séparation des caractéristiques (X) et de la cible à prédire (y)
X = df_ml[["temperature_f", "precipitations_pouces", "mois", "jour_semaine"]]
y = df_ml["total_locations"]

# Division en données d'entraînement (80%) et de test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# 3. AUTO-MACHINE LEARNING AVEC TPOT
# ---------------------------------------------------------
print("\n🤖 Lancement de TPOT (Recherche du meilleur modèle)...")
print("⚠️ Cela peut prendre plusieurs minutes. Laissez l'IA travailler !")

# Configuration légère pour commencer (generations=5, population=20)
# Dans un projet final, on augmente ces valeurs pour une recherche plus approfondie
tpot = TPOTRegressor(
    generations=5, 
    population_size=20, 
    verbosity=2, 
    random_state=42, 
    n_jobs=-1 # Utilise tous les cœurs de votre processeur
)

tpot.fit(X_train, y_train)

# ---------------------------------------------------------
# 4. RÉSULTATS ET EXPORTATION
# ---------------------------------------------------------
print("\n✅ Entraînement terminé !")

# Évaluation sur les données de test (que le modèle n'a jamais vues)
predictions = tpot.predict(X_test)
score_r2 = r2_score(y_test, predictions)

print("-" * 50)
print(f"🎯 Score R² du meilleur modèle : {score_r2:.2f}")
print("(Un score proche de 1.0 est excellent, proche de 0 est mauvais)")
print("-" * 50)

# TPOT génère automatiquement le code Python du meilleur modèle trouvé !
fichier_export = "src/meilleur_modele_pipeline.py"
tpot.export(fichier_export)
print(f"💾 Le code source du meilleur pipeline a été sauvegardé dans : {fichier_export}")
