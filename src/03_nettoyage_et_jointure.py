import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, when

print("🚀 Démarrage de la session PySpark...")
spark = SparkSession.builder \
    .appName("CitiBike_Meteo_Join") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()

# ---------------------------------------------------------
# 1. CHARGEMENT DES DONNÉES
# ---------------------------------------------------------
print("📂 Chargement des bases de données Parquet...")
df_velos = spark.read.parquet("data/citibike_db")
df_meteo = spark.read.parquet("data/meteo_db")

# ---------------------------------------------------------
# 2. NETTOYAGE DES VÉLOS
# ---------------------------------------------------------
print("🧹 Nettoyage des données vélos...")

# Identification de la colonne de départ (Citi Bike a changé le nom de "starttime" à "started_at" au fil des ans)
col_debut = "starttime" if "starttime" in df_velos.columns else "started_at"

# Extraction de la date pure (AAAA-MM-JJ) pour pouvoir faire la correspondance avec la météo
df_velos_clean = df_velos.withColumn("date_trajet", to_date(col(col_debut)))

# Suppression des "faux" trajets (moins de 60 secondes = souvent un vélo défectueux reposé)
if "tripduration" in df_velos_clean.columns:
    df_velos_clean = df_velos_clean.filter(col("tripduration") > 60)

# ---------------------------------------------------------
# 3. NETTOYAGE DE LA MÉTÉO
# ---------------------------------------------------------
print("🌦️ Nettoyage des données météorologiques...")

# La NOAA utilise 9999.9 pour les températures manquantes et 99.99 pour la pluie manquante.
# On remplace ces valeurs par du vide (Null) ou par 0 pour la pluie.
df_meteo_clean = df_meteo.select(
    to_date(col("DATE")).alias("date_meteo"),
    when(col("TEMP") == 9999.9, None).otherwise(col("TEMP")).alias("temperature_f"),
    when(col("PRCP") == 99.99, 0.0).otherwise(col("PRCP")).alias("precipitations_pouces")
)

# ---------------------------------------------------------
# 4. JOINTURE (MERGE)
# ---------------------------------------------------------
print("🔗 Croisement des trajets avec la météo du jour...")

# Jointure interne (Inner Join) sur la date
df_final = df_velos_clean.join(
    df_meteo_clean, 
    df_velos_clean.date_trajet == df_meteo_clean.date_meteo, 
    "inner"
)

# On retire la colonne date en double pour garder un tableau propre
df_final = df_final.drop("date_meteo")

# ---------------------------------------------------------
# 5. SAUVEGARDE FINALE
# ---------------------------------------------------------
dossier_sortie = "../data/dataset_final_modelisation"
print(f"💾 Sauvegarde du jeu de données unifié dans {dossier_sortie}...")

df_final.write.mode("overwrite").parquet(dossier_sortie)

print("🎉 OPÉRATION TERMINÉE ! Le jeu de données est prêt pour l'exploration et le Machine Learning.")
