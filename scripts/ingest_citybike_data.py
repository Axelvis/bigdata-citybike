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
dossier_temp = '../data/temp_csv'      # Le "sas" de décompression
dossier_final = '../data/citibike_db'  # La base de données finale

# On s'assure de partir sur un dossier temporaire totalement vide
if os.path.exists(dossier_temp):
    shutil.rmtree(dossier_temp)
os.makedirs(dossier_temp, exist_ok=True)
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
# On boucle sur la totalité des fichiers sans aucune limite
for index, fichier_zip in enumerate(fichiers_zip, start=1):
    nom_zip = fichier_zip.split('/')[-1]
    chemin_zip_local = os.path.join(dossier_temp, nom_zip)
    
    print(f"[{index}/{total_fichiers}] ⬇️ Téléchargement de {nom_zip}...")
    
    # 1. Téléchargement rapide d'un seul bloc
    fs.get(fichier_zip, chemin_zip_local)
    
    print(f"[{index}/{total_fichiers}] 📦 Décompression locale...")
    
    # 2. Extraction des CSV
    with zipfile.ZipFile(chemin_zip_local, 'r') as z:
        vrais_csv = [
            nom for nom in z.namelist() 
            if nom.endswith('.csv') and not nom.startswith('__MACOSX') and not nom.split('/')[-1].startswith('._')
        ]
        for csv_file in vrais_csv:
            z.extract(csv_file, dossier_temp)
            
    print(f"[{index}/{total_fichiers}] ⚡ Traitement PySpark...")
    
    # 3. Lecture PySpark 
    # recursiveFileLookup=true permet de trouver les CSV même s'ils sont dans des sous-dossiers du ZIP
    df_spark = spark.read.option("recursiveFileLookup", "true") \
                         .csv(dossier_temp, header=True, inferSchema=True)
    
    # 4. Ajout à la base Parquet
    # L'option mergeSchema=true est ajoutée en sécurité au cas où Citi Bike aurait 
    # rajouté ou modifié des colonnes entre 2013 et 2024.
    df_spark.write.mode("append").option("mergeSchema", "true").parquet(dossier_final)
    
    print(f"[{index}/{total_fichiers}] 🧹 Nettoyage du disque...")
    
    # 5. Destruction totale du contenu du dossier temporaire (ZIP + CSV)
    shutil.rmtree(dossier_temp)
    os.makedirs(dossier_temp, exist_ok=True)
    
    print(f"✅ Fichier {nom_zip} sécurisé en Parquet !\n")

print("🎉 PIPELINE GLOBAL TERMINÉ ! Toutes vos données sont prêtes.")
