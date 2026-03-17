import s3fs
import os
import shutil
from pyspark.sql import SparkSession

# ---------------------------------------------------------
# 0. INITIALISATION DE SPARK
# ---------------------------------------------------------
print("🚀 Démarrage de la session PySpark pour la Météo...")
spark = SparkSession.builder \
    .appName("Meteo_Pipeline_NYC") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()
print("✅ Session PySpark prête !")
print("-" * 50)

# ---------------------------------------------------------
# 1. CONFIGURATION DES DOSSIERS ET PARAMÈTRES
# ---------------------------------------------------------
dossier_temp = '../data/temp_meteo'
dossier_final = '../data/meteo_db'

if os.path.exists(dossier_temp):
    shutil.rmtree(dossier_temp)
os.makedirs(dossier_temp, exist_ok=True)
os.makedirs(dossier_final, exist_ok=True)

# Identifiants NOAA pour la station de Central Park, NY
station_id = "72505394728"
annees = range(2013, 2025)  # De 2013 à 2024 inclus

fs = s3fs.S3FileSystem(anon=True)

print(f"Total d'années à traiter : {len(annees)} (de {annees[0]} à {annees[-1]})")
print("-" * 50)

# ---------------------------------------------------------
# 2. LA BOUCLE GLOBALE (Pipeline)
# ---------------------------------------------------------
for annee in annees:
    fichier_s3 = f"noaa-gsod-pds/{annee}/{station_id}.csv"
    chemin_local = os.path.join(dossier_temp, f"meteo_nyc_{annee}.csv")
    
    if fs.exists(fichier_s3):
        print(f"[{annee}] ⬇️ Téléchargement des relevés météo...")
        fs.get(fichier_s3, chemin_local)
        
        print(f"[{annee}] ⚡ Traitement PySpark...")
        # Spark lit spécifiquement ce fichier CSV local
        df_spark = spark.read.csv(chemin_local, header=True, inferSchema=True)
        
        # Nettoyage des espaces cachés dans les en-têtes de colonnes de la NOAA
        colonnes_propres = [col.strip() for col in df_spark.columns]
        df_spark = df_spark.toDF(*colonnes_propres)
        
        # Ajout direct en Parquet
        df_spark.write.mode("append").parquet(dossier_final)
        
        os.remove(chemin_local)
        
        print(f"✅ Année {annee} sécurisée en Parquet !\n")
    else:
        print(f"⚠️ Aucune donnée trouvée pour l'année {annee} à Central Park.\n")

# Nettoyage final
shutil.rmtree(dossier_temp)

print("🎉 PIPELINE MÉTÉO TERMINÉ ! Vos données climatiques sont prêtes.")
