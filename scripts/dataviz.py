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
# 1. CHARGEMENT ET STANDARDISATION
# ---------------------------------------------------------
print("Chargement du dataset Parquet...")
df_final = spark.read.parquet("data/dataset_final_modelisation")

# Standardisation de la colonne utilisateur
# Citi Bike utilise "usertype" avant 2021 et "member_casual" ensuite.
col_user = "member_casual" if "member_casual" in df_final.columns else "usertype"
df_final = df_final.withColumnRenamed(col_user, "type_utilisateur")

# ---------------------------------------------------------
# 2. AGREGATIONS DISTRIBUEES
# ---------------------------------------------------------
print("Calcul des agregations par les Executors Spark...")

# Analyse 1 : Impact global de la temperature et de la pluie par jour
df_meteo_impact = df_final.groupBy("date_trajet", "temperature_f", "precipitations_pouces") \
    .count() \
    .withColumnRenamed("count", "total_trajets")

# Analyse 2 : Resilience face a la pluie
df_pluie = df_final.withColumn(
    "temps_pluvieux", 
    when(col("precipitations_pouces") > 0.1, "Pluie").otherwise("Sec")
)

df_resilience = df_pluie.groupBy("date_trajet", "temps_pluvieux", "type_utilisateur") \
    .count() \
    .withColumnRenamed("count", "trajets_par_type")

# ---------------------------------------------------------
# 3. RAPATRIEMENT SUR LE DRIVER
# ---------------------------------------------------------
print("Rapatriement des donnees agregees vers le Driver (toPandas)...")
pdf_meteo = df_meteo_impact.toPandas()
pdf_resilience = df_resilience.toPandas()

# ---------------------------------------------------------
# 4. VISUALISATIONS (Seaborn & Matplotlib)
# ---------------------------------------------------------
print("Generation des graphiques...")

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Graphique 1 : Nuage de points
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

# Graphique 2 : Boites a moustaches
sns.boxplot(
    data=pdf_resilience, 
    x="type_utilisateur", 
    y="trajets_par_type", 
    hue="temps_pluvieux", 
    palette={"Sec": "#2ecc71", "Pluie": "#e74c3c"},
    ax=axes[1]
)
axes[1].set_title("Resilience face a la pluie selon le profil", fontsize=14)
axes[1].set_xlabel("Profil d'utilisateur", fontsize=12)
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
