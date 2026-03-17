import s3fs
import zipfile
import os
import shutil
import time
from pyspark.sql import SparkSession

# ---------------------------------------------------------
# 0. INITIALISATION DE SPARK
# ---------------------------------------------------------
print("Demarrage de la session PySpark pour le Benchmark...")
spark = SparkSession.builder \
    .appName("CitiBike_Benchmark") \
    .getOrCreate()

# ---------------------------------------------------------
# 1. PREPARATION DES DONNEES DE TEST (1 SEUL MOIS)
# ---------------------------------------------------------
dossier_csv = 'data/benchmark_csv'
dossier_parquet = 'data/benchmark_parquet'
fichier_test_s3 = 'tripdata/202401-citibike-tripdata.zip'
chemin_zip_local = 'data/benchmark_temp.zip'

# Nettoyage initial
for d in [dossier_csv, dossier_parquet]:
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)

print("Telechargement d'un mois de donnees (Janvier 2024) pour le test...")
fs = s3fs.S3FileSystem(anon=True)
fs.get(fichier_test_s3, chemin_zip_local)

print("Decompression en CSV...")
with zipfile.ZipFile(chemin_zip_local, 'r') as z:
    for nom in z.namelist():
        if nom.endswith('.csv') and '__MACOSX' not in nom:
            z.extract(nom, dossier_csv)
os.remove(chemin_zip_local)

print("Creation de la copie exacte en format Parquet...")
# On lit le CSV et on l'ecrit en Parquet pour avoir une base de comparaison stricte
df_csv_prep = spark.read.csv(dossier_csv, header=True, inferSchema=True)
df_csv_prep.write.mode("overwrite").parquet(dossier_parquet)

# ---------------------------------------------------------
# 2. LE BENCHMARK
# ---------------------------------------------------------
print("\n" + "="*50)
print("DEBUT DU BENCHMARK ANALYTIQUE")
print("Requete : Compter le nombre de trajets par type d'utilisateur")
print("="*50)

# On force Spark a vider son cache pour que le test soit juste
spark.catalog.clearCache()

# Test 1 : Lecture depuis le CSV
print("\n[1] Lancement du traitement sur le format CSV...")
df_csv = spark.read.csv(dossier_csv, header=True, inferSchema=True)

start_time_csv = time.time()
# L'action collect() declenche le calcul
resultat_csv = df_csv.groupBy("member_casual").count().collect()
duree_csv = time.time() - start_time_csv

print(f"Temps d'execution CSV : {duree_csv:.2f} secondes")

# On vide le cache a nouveau
spark.catalog.clearCache()

# Test 2 : Lecture depuis le Parquet
print("\n[2] Lancement du traitement sur le format Parquet...")
df_parquet = spark.read.parquet(dossier_parquet)

start_time_parquet = time.time()
resultat_parquet = df_parquet.groupBy("member_casual").count().collect()
duree_parquet = time.time() - start_time_parquet

print(f"Temps d'execution Parquet : {duree_parquet:.2f} secondes")

# ---------------------------------------------------------
# 3. RESULTATS ET NETTOYAGE
# ---------------------------------------------------------
print("\n" + "="*50)
print("CONCLUSION DU BENCHMARK")
print("="*50)
if duree_parquet > 0:
    acceleration = duree_csv / duree_parquet
    print(f"Le format Parquet a ete {acceleration:.1f} fois plus rapide que le CSV !")

print("\nNettoyage des fichiers de test...")
shutil.rmtree(dossier_csv)
shutil.rmtree(dossier_parquet)

spark.stop()
