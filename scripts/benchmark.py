import s3fs
import zipfile
import os
import shutil
import time
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession

# ---------------------------------------------------------
# 0. INITIALISATION DE SPARK
# ---------------------------------------------------------
print("🚀 Demarrage de la session PySpark pour le Benchmark Scalable...")
spark = SparkSession.builder \
    .appName("CitiBike_Scalability_Benchmark") \
    .getOrCreate()

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------
dossier_csv = 'data/benchmark_csv'
dossier_parquet = 'data/benchmark_parquet'
chemin_zip_local = 'data/benchmark_temp.zip'

# On va tester sur le premier trimestre 2024 (ajout progressif)
fichiers_s3_a_tester = [
    'tripdata/202401-citibike-tripdata.zip',
    'tripdata/202402-citibike-tripdata.zip',
    'tripdata/202403-citibike-tripdata.zip'
    # Vous pouvez en ajouter plus, mais attention au temps d'exécution total !
]

# Nettoyage initial strict
for d in [dossier_csv, dossier_parquet]:
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)

fs = s3fs.S3FileSystem(anon=True)
resultats_benchmark = []

# ---------------------------------------------------------
# 2. BOUCLE D'ACCUMULATION ET BENCHMARK
# ---------------------------------------------------------
print("\n" + "="*60)
print("DEBUT DU BENCHMARK DE SCALABILITE")
print("="*60)

for mois_index, fichier_s3 in enumerate(fichiers_s3_a_tester, start=1):
    print(f"\n--- ETAPE {mois_index} : Test avec {mois_index} mois de donnees cumulées ---")
    
    # 2.1 Téléchargement et ajout des nouveaux CSV
    print(f"⬇️ Telechargement de {fichier_s3}...")
    fs.get(fichier_s3, chemin_zip_local)
    
    print("📦 Decompression (accumulation avec les mois precedents)...")
    with zipfile.ZipFile(chemin_zip_local, 'r') as z:
        for nom in z.namelist():
            if nom.endswith('.csv') and '__MACOSX' not in nom:
                z.extract(nom, dossier_csv)
    os.remove(chemin_zip_local)
    
    # 2.2 Création du Parquet équivalent
    print("🔄 Creation de la copie exacte en format Parquet...")
    df_csv_prep = spark.read.csv(dossier_csv, header=True, inferSchema=True)
    df_csv_prep.write.mode("overwrite").parquet(dossier_parquet)
    
    # 2.3 BENCHMARK CSV
    spark.catalog.clearCache()
    print("⏱️  Mesure du temps CSV...")
    df_csv = spark.read.csv(dossier_csv, header=True, inferSchema=True)
    start_csv = time.time()
    df_csv.groupBy("member_casual").count().collect()
    temps_csv = time.time() - start_csv
    
    # 2.4 BENCHMARK PARQUET
    spark.catalog.clearCache()
    print("⏱️  Mesure du temps Parquet...")
    df_parquet = spark.read.parquet(dossier_parquet)
    start_parquet = time.time()
    df_parquet.groupBy("member_casual").count().collect()
    temps_parquet = time.time() - start_parquet
    
    print(f"✅ Resultat Étape {mois_index} | CSV : {temps_csv:.2f}s | Parquet : {temps_parquet:.2f}s")
    
    # Sauvegarde des résultats
    resultats_benchmark.append({
        "Mois_Cumules": mois_index,
        "Temps_CSV_secondes": temps_csv,
        "Temps_Parquet_secondes": temps_parquet
    })

# ---------------------------------------------------------
# 3. VISUALISATION DES RESULTATS
# ---------------------------------------------------------
print("\n" + "="*60)
print("GENERATION DES COURBES DE PERFORMANCE")
print("="*60)

# Transformation en DataFrame Pandas pour le graphique
df_results = pd.DataFrame(resultats_benchmark)

plt.figure(figsize=(10, 6))
plt.plot(df_results["Mois_Cumules"], df_results["Temps_CSV_secondes"], marker='o', color='red', linewidth=2, label='Format CSV')
plt.plot(df_results["Mois_Cumules"], df_results["Temps_Parquet_secondes"], marker='s', color='green', linewidth=2, label='Format Parquet')

plt.title("Évolution du temps de calcul selon le volume de données", fontsize=14)
plt.xlabel("Volume de données (Nombre de mois cumulés)", fontsize=12)
plt.ylabel("Temps d'exécution (en secondes)", fontsize=12)
plt.xticks(df_results["Mois_Cumules"])
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

fichier_graphique = "data/benchmark_scalabilite.png"
plt.savefig(fichier_graphique)
print(f"📊 Graphique généré et sauvegardé sous : {fichier_graphique}")
plt.show()

# Nettoyage final
print("\nNettoyage des fichiers temporaires massifs...")
shutil.rmtree(dossier_csv)
shutil.rmtree(dossier_parquet)

spark.stop()
print("🎉 Benchmark terminé !")
