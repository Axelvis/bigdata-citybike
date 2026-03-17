import s3fs
import zipfile
import os
import shutil
from pyspark.sql import SparkSession

# ---------------------------------------------------------
# 0. INITIALISATION DE SPARK
# ---------------------------------------------------------
print("🚀 Démarrage de la session PySpark locale...")
spark = SparkSession.builder \
    .appName("CitiBike_Pipeline_Global") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()
print("✅ Session PySpark prête !")
print("-" * 50)

# ---------------------------------------------------------
# 1. CONFIGURATION DES DOSSIERS
# ---------------------------------------------------------
# CRITIQUE : On sépare physiquement le ZIP des CSV pour ne pas perturber Spark
dossier_zip = 'data/temp_zip'       # Accueil du fichier téléchargé
dossier_csv = 'data/temp_csv'       # Accueil des fichiers décompressés
dossier_final = 'data/citibike_db'  # Base de données finale

# Nettoyage et création des dossiers
for d in [dossier_zip, dossier_csv]:
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)
os.makedirs(dossier_final, exist_ok=True)

# ---------------------------------------------------------
# 2. LISTAGE DES FICHIERS SUR S3
# ---------------------------------------------------------
fs = s3fs.S3FileSystem(anon=True)
fichiers_zip = [f for f in fs.ls('tripdata') if f.endswith('.zip')]

total_fichiers = len(fichiers_zip)
print(f"Total de fichiers ZIP à traiter : {total_fichiers}")
print("-" * 50)

# ---------------------------------------------------------
# 3. LA BOUCLE GLOBALE (Pipeline)
# ---------------------------------------------------------
for index, fichier_zip in enumerate(fichiers_zip, start=1):
    nom_zip = fichier_zip.split('/')[-1]
    chemin_zip_local = os.path.join(dossier_zip, nom_zip)
    
    print(f"[{index}/{total_fichiers}] ⬇️ Téléchargement de {nom_zip}...")
    fs.get(fichier_zip, chemin_zip_local)
    
    print(f"[{index}/{total_fichiers}] 📦 Décompression locale...")
    with zipfile.ZipFile(chemin_zip_local, 'r') as z:
        vrais_csv = [
            nom for nom in z.namelist() 
            if nom.endswith('.csv') and not nom.startswith('__MACOSX') and not nom.split('/')[-1].startswith('._')
        ]
        for csv_file in vrais_csv:
            # On extrait uniquement dans le dossier réservé aux CSV
            z.extract(csv_file, dossier_csv)
            
    print(f"[{index}/{total_fichiers}] ⚡ Traitement PySpark...")
    # Spark ne scannera QUE les fichiers texte de ce dossier
    df_spark = spark.read.option("recursiveFileLookup", "true") \
                         .csv(dossier_csv, header=True, inferSchema=True)
    
    df_spark.write.mode("append").option("mergeSchema", "true").parquet(dossier_final)
    
    print(f"[{index}/{total_fichiers}] 🧹 Nettoyage du disque...")
    # On vide les sas pour faire place nette à la prochaine archive
    shutil.rmtree(dossier_zip)
    shutil.rmtree(dossier_csv)
    os.makedirs(dossier_zip, exist_ok=True)
    os.makedirs(dossier_csv, exist_ok=True)
    
    print(f"✅ Fichier {nom_zip} sécurisé en Parquet !\n")

print("🎉 PIPELINE GLOBAL VÉLOS TERMINÉ ! Toutes vos données sont prêtes.")
