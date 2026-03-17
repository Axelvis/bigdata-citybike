import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, hour

print("🚀 Démarrage de la session PySpark...")
spark = SparkSession.builder \
    .appName("CitiBike_Meteo_Horaire_Join") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()

# ---------------------------------------------------------
# 1. CHARGEMENT DES DONNÉES
# ---------------------------------------------------------
print("📂 Chargement des bases de données Parquet...")
df_velos = spark.read.parquet("data/citibike_db")

# Attention : On pointe désormais vers le nouveau dossier horaire !
df_meteo = spark.read.parquet("data/meteo_horaire_db")

# ---------------------------------------------------------
# 2. PRÉPARATION DES VÉLOS (Date + Heure)
# ---------------------------------------------------------
print("🧹 Extraction des marqueurs temporels des trajets...")

col_debut = "starttime" if "starttime" in df_velos.columns else "started_at"

# NOUVEAUTÉ : On extrait la date ET l'heure (de 0 à 23)
df_velos_clean = df_velos.withColumn("date_trajet", to_date(col(col_debut))) \
                         .withColumn("heure_trajet", hour(col(col_debut)))

# Filtre de propreté sur les trajets fantômes
if "tripduration" in df_velos_clean.columns:
    df_velos_clean = df_velos_clean.filter(col("tripduration") > 60)

# ---------------------------------------------------------
# 3. PRÉPARATION DE LA MÉTÉO
# ---------------------------------------------------------
print("🌦️ Formatage des données météorologiques...")

# Les données ont déjà été nettoyées par Pandas, on renomme juste 
# les colonnes pour être très clair sur les unités (Celsius, mm, km/h)
df_meteo_clean = df_meteo.select(
    col("date_meteo"),
    col("heure_meteo"),
    col("temp").alias("temperature_c"),
    col("prcp").alias("precipitations_mm"),
    col("wspd").alias("vent_kmh")
)

# ---------------------------------------------------------
# 4. JOINTURE ULTRA-PRÉCISE (MERGE)
# ---------------------------------------------------------
print("🔗 Croisement des trajets avec la météo HORAIRE...")

# NOUVEAUTÉ : Jointure sur 2 conditions (Date == Date ET Heure == Heure)
df_final = df_velos_clean.join(
    df_meteo_clean, 
    (df_velos_clean.date_trajet == df_meteo_clean.date_meteo) & 
    (df_velos_clean.heure_trajet == df_meteo_clean.heure_meteo), 
    "inner"
)

# On retire les colonnes météo en double pour garder un tableau propre
df_final = df_final.drop("date_meteo", "heure_meteo")

# ---------------------------------------------------------
# 5. SAUVEGARDE FINALE
# ---------------------------------------------------------
# On change le nom du dossier de sortie pour ne pas écraser votre ancienne base quotidienne
dossier_sortie = "data/dataset_horaire_final"
print(f"💾 Sauvegarde du jeu de données unifié dans {dossier_sortie}...")

df_final.write.mode("overwrite").parquet(dossier_sortie)

print("🎉 OPÉRATION TERMINÉE ! Le jeu de données Haute Précision est prêt.")
