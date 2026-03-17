import s3fs
import zipfile
import os
import shutil
from pyspark.sql import SparkSession

# ---------------------------------------------------------
# 0. INITIALISATION DE SPARK
# ---------------------------------------------------------
print("Demarrage de la session PySpark locale...")
spark = SparkSession.builder \
    .appName("CitiBike_Pipeline_Global_Filtre") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()
print("Session PySpark prete !")
print("-" * 50)

# ---------------------------------------------------------
# 1. CONFIGURATION DES DOSSIERS
# ---------------------------------------------------------
dossier_zip = 'data/temp_zip'       
dossier_csv = 'data/temp_csv'       
dossier_final = 'data/citibike_db'  

for d in [dossier_zip, dossier_csv]:
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)
os.makedirs(dossier_final, exist_ok=True)

# ---------------------------------------------------------
# 2. LISTAGE ET FILTRAGE DES FICHIERS SUR S3 (2017 à 2025)
# ---------------------------------------------------------
fs = s3fs.S3FileSystem(anon=True)
tous_les_fichiers = fs.ls('tripdata')

# Création d'une liste contenant les années sous forme de texte ['2017', '2018', ..., '2025']
annees_cibles = [str(annee) for annee in range(2017, 2026)]

fichiers_zip = []
for f in tous_les_fichiers:
    if f.endswith('.zip'):
        nom_fichier = f.split('/')[-1]
        # On conserve le fichier uniquement si une des années cibles est dans son nom
        if any(annee in nom_fichier for annee in annees_cibles):
            fichiers_zip.append(f)

total_fichiers = len(fichiers_zip)
print(f"Total de fichiers ZIP a traiter (2017-2025) : {total_fichiers}")
print("-" * 50)

# ---------------------------------------------------------
# 3. LA BOUCLE GLOBALE (Pipeline)
# ---------------------------------------------------------
for index, fichier_zip in enumerate(fichiers_zip, start=1):
    nom_zip = fichier_zip.split('/')[-1]
    chemin_zip_local = os.path.join(dossier_zip, nom_zip)
    
    print(f"[{index}/{total_fichiers}] Telechargement de {nom_zip}...")
    fs.get(fichier_zip, chemin_zip_local)
    
    print(f"[{index}/{total_fichiers}] Decompression locale...")
    with zipfile.ZipFile(chemin_zip_local, 'r') as z:
        for nom in z.namelist():
            if nom.endswith('.csv') and '__MACOSX' not in nom and not nom.split('/')[-1].startswith('._'):
                z.extract(nom, dossier_csv)
                chemin_extrait = os.path.join(dossier_csv, nom)
                
                # Patch Anti-Anomalie Citi Bike (Faux CSV)
                if zipfile.is_zipfile(chemin_extrait):
                    print(f"   Anomalie detectee : {nom} est un ZIP deguise ! Double extraction...")
                    with zipfile.ZipFile(chemin_extrait, 'r') as z_cache:
                        z_cache.extractall(dossier_csv)
                    os.remove(chemin_extrait)

    print(f"[{index}/{total_fichiers}] Traitement PySpark...")
    
    # On force Spark à ne lire que les fichiers qui ne sont pas des ZIP déguisés
    fichiers_a_lire = []
    for root, dirs, files in os.walk(dossier_csv):
        for f in files:
            chemin_vrai = os.path.join(root, f)
            if f.endswith('.csv') and not f.startswith('._') and '__MACOSX' not in root:
                if not zipfile.is_zipfile(chemin_vrai):
                    fichiers_a_lire.append(chemin_vrai)

    if fichiers_a_lire:
        df_spark = spark.read.csv(fichiers_a_lire, header=True, inferSchema=True)
        df_spark.write.mode("append").option("mergeSchema", "true").parquet(dossier_final)
        print(f"Fichier {nom_zip} securise en Parquet !\n")
    else:
        print(f"Aucun CSV valide trouve dans {nom_zip} !\n")
    
    print(f"[{index}/{total_fichiers}] Nettoyage du disque...")
    shutil.rmtree(dossier_zip)
    shutil.rmtree(dossier_csv)
    os.makedirs(dossier_zip, exist_ok=True)
    os.makedirs(dossier_csv, exist_ok=True)

print("PIPELINE GLOBAL VELOS TERMINE ! Toutes vos donnees sont pretes.")
