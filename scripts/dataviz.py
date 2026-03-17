import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

# ---------------------------------------------------------
# 0. INITIALISATION DE SPARK
# ---------------------------------------------------------
print("Demarrage de la session PySpark pour la Dataviz...")
spark = SparkSession.builder \
    .appName("CitiBike_Dataviz") \
    .getOrCreate()

# ---------------------------------------------------------
# 1. CHARGEMENT DU DATASET FINAL (Optimisation Parquet)
# ---------------------------------------------------------
print("Chargement du dataset Parquet...")
# Le format Parquet etant oriente colonnes, Spark ne lira sur le disque 
# que les colonnes explicitement utilisees dans les transformations suivantes 
# (projection pushdown) .
df_final = spark.read.parquet("data/dataset_final_modelisation")

# ---------------------------------------------------------
# 2. AGREGATIONS DISTRIBUEES (Lazy Evaluation & Catalyst)
# ---------------------------------------------------------
print("Calcul des agregations par les Executors Spark...")

# Analyse 1 : Impact global de la temperature et de la pluie par jour
# Ces transformations sont declaratives. Le moteur Catalyst va les optimiser 
# avant de generer le plan d'execution physique[cite: 149, 155, 156].
df_meteo_impact = df_final.groupBy("date_trajet", "temperature_f", "precipitations_pouces") \
    .count() \
    .withColumnRenamed("count", "total_trajets")

# Analyse 2 : Resilience face a la pluie (Abonnes vs Occasionnels)
# On cree d'abord une colonne binaire pour la pluie
df_pluie = df_final.withColumn(
    "temps_pluvieux", 
    when(col("precipitations_pouces") > 0.1, "Pluie").otherwise("Sec")
)

df_resilience = df_pluie.groupBy("date_trajet", "temps_pluvieux", "member_casual") \
    .count() \
    .withColumnRenamed("count", "trajets_par_type")

# ---------------------------------------------------------
# 3. RAPATRIEMENT SUR LE DRIVER (Action)
# ---------------------------------------------------------
print("Rapatriement des donnees agregees vers le Driver (toPandas)...")
# L'appel a toPandas() est l'Action qui declenche reellement les calculs[cite: 88, 90, 147].
# Seul le resultat final (quelques milliers de lignes) est charge en memoire vive locale.
pdf_meteo = df_meteo_impact.toPandas()
pdf_resilience = df_resilience.toPandas()

# ---------------------------------------------------------
# 4. VISUALISATIONS (Seaborn & Matplotlib)
# ---------------------------------------------------------
print("Generation des graphiques...")

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Graphique 1 : Nuage de points (Temperature vs Nombre de trajets)
sns.scatterplot(
    data=pdf_meteo, 
    x="temperature_f", 
    y="total_trajets", 
    alpha=0.5, 
    color="#3498db", 
    ax=axes[0]
)
axes[0].set_title("Impact de la temperature sur le volume de locations", fontsize=14)
axes[0].set_xlabel("Temperature moyenne quotidienne (Fahrenheit)", fontsize=12)
axes[0].set_ylabel("Nombre total de trajets", fontsize=12)

# Graphique 2 : Boites à moustaches (Resilience à la pluie par type d'utilisateur)
sns.boxplot(
    data=pdf_resilience, 
    x="member_casual", 
    y="trajets_par_type", 
    hue="temps_pluvieux", 
    palette={"Sec": "#2ecc71", "Pluie": "#e74c3c"},
    ax=axes[1]
)
axes[1].set_title("Resilience face a la pluie : Abonnes vs Occasionnels", fontsize=14)
axes[1].set_xlabel("Type d'utilisateur (Casual = Occasionnel, Member = Abonne)", fontsize=12)
axes[1].set_ylabel("Volume de trajets par jour", fontsize=12)

plt.tight_layout()
plt.savefig("data/graphiques_meteo.png")
print("Graphiques sauvegardes sous 'data/graphiques_meteo.png'. Affichage a l'ecran...")
plt.show()

# ---------------------------------------------------------
# 5. FERMETURE DE LA SESSION
# ---------------------------------------------------------
spark.stop()
print("Operation terminee avec succes.")
