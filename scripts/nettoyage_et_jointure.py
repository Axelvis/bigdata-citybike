import os
import glob
from functools import reduce
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, hour, coalesce, unix_timestamp

print("🚀 Démarrage de la session PySpark...")
spark = SparkSession.builder \
    .appName("CitiBike_Meteo_Horaire_Join") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()

# ---------------------------------------------------------
# 1. LECTURE INTELLIGENTE ET FUSION (LE VRAI BLINDAGE)
# ---------------------------------------------------------
print("📂 Analyse et harmonisation des fichiers Parquet (anti-corruption)...")

# On liste tous les fichiers parquet physiquement présents dans ton dossier
fichiers_parquet = glob.glob("data/citibike_db/*.parquet")

dfs = []
for fichier in fichiers_parquet:
    # Spark lit le fichier avec son VRAI type interne (INT, TIMESTAMP, etc.)
    df_temp = spark.read.parquet(fichier)
    
    # On convertit tout de suite proprement en texte AVANT de les mélanger
    for c in df_temp.columns:
        df_temp = df_temp.withColumn(c, col(c).cast("string"))
        
    dfs.append(df_temp)

# Fusion globale en autorisant les colonnes manquantes entre les anciennes et nouvelles années
print("🔄 Empilage des historiques en cours (ça peut prendre un instant)...")
df_velos = reduce(lambda df1, df2: df1.unionByName(df2, allowMissingColumns=True), dfs)


# ---------------------------------------------------------
# 2. CHARGEMENT DE LA MÉTÉO
# ---------------------------------------------------------
df_meteo = spark.read.parquet("data/meteo_horaire_multi_db")


# ---------------------------------------------------------
# 3. PRÉPARATION DES VÉLOS
# ---------------------------------------------------------
print("🧹 Extraction et unification des marqueurs temporels...")

# Maintenant qu'on a un beau texte formaté "2021-01-01 12:00:00", le cast en timestamp marchera !
df_velos_clean = df_velos.withColumn("debut_unifie", coalesce(col("starttime"), col("started_at")).cast("timestamp")) \
                         .withColumn("fin_unifie", coalesce(col("stoptime"), col("ended_at")).cast("timestamp"))

df_velos_clean = df_velos_clean.withColumn("date_trajet", to_date(col("debut_unifie"))) \
                               .withColumn("heure_trajet", hour(col("debut_unifie")))

# Calcul de la durée du trajet en secondes
df_velos_clean = df_velos_clean.withColumn("duree_secondes",
    coalesce(
        col("tripduration").cast("int"),
        (unix_timestamp(col("fin_unifie")) - unix_timestamp(col("debut_unifie"))).cast("int")
    )
)

# Filtre de propreté
df_velos_clean = df_velos_clean.filter(col("duree_secondes") > 60)


# ---------------------------------------------------------
# 4. PRÉPARATION DE LA MÉTÉO
# ---------------------------------------------------------
print("🌦️ Formatage des données météorologiques...")
df_meteo_clean = df_meteo.select(
    col("date_meteo"),
    col("heure_meteo"),
    col("temp").alias("temperature_c"),
    col("prcp").alias("precipitations_mm"),
    col("wspd").alias("vent_kmh")
)


# ---------------------------------------------------------
# 5. JOINTURE
# ---------------------------------------------------------
print("🔗 Croisement des trajets avec la météo HORAIRE...")
df_final = df_velos_clean.join(
    df_meteo_clean, 
    (df_velos_clean.date_trajet == df_meteo_clean.date_meteo) & 
    (df_velos_clean.heure_trajet == df_meteo_clean.heure_meteo), 
    "inner"
)

# Nettoyage des colonnes en double
df_final = df_final.drop("date_meteo", "heure_meteo")


# ---------------------------------------------------------
# 6. SAUVEGARDE FINALE
# ---------------------------------------------------------
dossier_sortie = "data/dataset_horaire_final"
print(f"💾 Sauvegarde du jeu de données unifié dans {dossier_sortie}...")

df_final.write.mode("overwrite").parquet(dossier_sortie)

print("🎉 OPÉRATION TERMINÉE ! Le jeu de données Haute Précision est prêt.")
