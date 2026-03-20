import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, coalesce, date_format, avg, count, dayofweek
from matplotlib.colors import LinearSegmentedColormap

# Configuration - Pour de plus beaux graphiques
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 100 # Meilleure résolution

# ---------------------------------------------------------
# 0. INITIALISATION DE SPARK
# ---------------------------------------------------------
print("🚀 Démarrage de la session PySpark pour la Dataviz Haute Précision...")
spark = SparkSession.builder \
    .appName("CitiBike_Dataviz_Expert") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()

# ---------------------------------------------------------
# 1. CHARGEMENT ET STANDARDISATION
# ---------------------------------------------------------
Dossier_In = "data/dataset_horaire_final"
print(f"📂 Chargement du dataset unifié depuis {Dossier_In}...")
df = spark.read.parquet(Dossier_In)

df = df.withColumn("type_utilisateur", 
    when(coalesce(col("member_casual"), col("usertype")).like("member"), "Subscriber")
    .when(coalesce(col("member_casual"), col("usertype")).like("Subscriber"), "Subscriber")
    .otherwise("Casual")
)

df = df.filter((col("duree_secondes") > 60) & (col("duree_secondes") < 7200))

# ---------------------------------------------------------
# 2. CALCUL DES AGRÉGATIONS (100% DISTRIBUÉ SUR SPARK)
# ---------------------------------------------------------
print("📊 Calcul des agrégations complexes par les Executors Spark...")

# --- A. Tendance Globale (Mois par Mois) ---
df_mensuel = df.withColumn("mois_annee", date_format(col("date_trajet"), "yyyy-MM")) \
    .withColumn("annee", date_format(col("date_trajet"), "yyyy")) \
    .groupBy("mois_annee", "annee", "type_utilisateur") \
    .agg(
        count("*").alias("total_trajets"),
        avg("duree_secondes").alias("duree_moy_sec")
    ).orderBy("mois_annee")

# --- B. Profil Horaire (Semaine vs Week-end) ---
df_horaire = df.withColumn("type_jour", 
        when(dayofweek(col("date_trajet")).isin(1, 7), "Week-end").otherwise("Semaine")
    ) \
    .groupBy("type_jour", "heure_trajet", "type_utilisateur") \
    .count() \
    .withColumnRenamed("count", "total_trajets")

# --- C. Impact Météo (CORRIGÉ : Moyenne Horaire pour éviter le biais climatique) ---
# Étape 1 : On compte combien de trajets ont eu lieu pour CHAQUE heure de l'historique
df_meteo_horaire = df.withColumn("temp_arrondie", col("temperature_c").cast("int")) \
    .withColumn("vent_arrondi", (col("vent_kmh") / 5).cast("int") * 5) \
    .filter(col("temperature_c").isNotNull()) \
    .groupBy("date_trajet", "heure_trajet", "temp_arrondie", "vent_arrondi") \
    .agg(count("*").alias("trajets_dans_l_heure"))

# Étape 2 : On fait la moyenne de ces volumes horaires pour chaque condition météo
df_meteo = df_meteo_horaire.groupBy("temp_arrondie", "vent_arrondi") \
    .agg(avg("trajets_dans_l_heure").alias("moyenne_trajets_par_heure"))

# --- D. Resilience Pluie (Comparaison Annuelle) ---
df_resilience = df.withColumn("annee", date_format(col("date_trajet"), "yyyy")) \
    .withColumn("is_raining", when(col("precipitations_mm") > 1.0, "Pluie").otherwise("Sec")) \
    .groupBy("annee", "is_raining", "type_utilisateur") \
    .count()

# ---------------------------------------------------------
# 3. RAPATRIEMENT SUR LE DRIVER (toPandas)
# ---------------------------------------------------------
print("📥 Transfert des résultats allégés vers Pandas pour la visualisation...")
pdf_mensuel = df_mensuel.toPandas()
pdf_horaire = df_horaire.toPandas()
pdf_meteo = df_meteo.toPandas()
pdf_resilience = df_resilience.toPandas()

pdf_horaire = pdf_horaire.sort_values(['type_jour', 'heure_trajet'])

# ---------------------------------------------------------
# 4. GÉNÉRATION DU TABLEAU DE BORD (Dashboard)
# ---------------------------------------------------------
print("🎨 Génération du Dashboard (6 graphiques)...")

fig, axes = plt.subplots(3, 2, figsize=(20, 24))
plt.subplots_adjust(hspace=0.3, wspace=0.2)
fig.suptitle("Citi Bike NYC : Analyse Profonde Intégrée (Trajets & Météo)", fontsize=24, fontweight='bold', y=0.98)

pal_user = {"Subscriber": "#3498db", "Casual": "#e67e22"}

# --- G1 : Évolution Mensuelle (Volume) ---
sns.lineplot(data=pdf_mensuel, x="mois_annee", y="total_trajets", hue="type_utilisateur", palette=pal_user, marker="o", ax=axes[0, 0], linewidth=2.5)
axes[0, 0].set_title("1. Évolution du volume mensuel de trajets", fontsize=16, fontweight='bold')
axes[0, 0].set_ylabel("Nombre total de trajets")
for ind, label in enumerate(axes[0, 0].get_xticklabels()):
    label.set_visible(ind % 4 == 0)
plt.setp(axes[0, 0].get_xticklabels(), rotation=45)

# --- G2 : Durée Moyenne des trajets ---
sns.barplot(data=pdf_mensuel, x="annee", y="duree_moy_sec", hue="type_utilisateur", palette=pal_user, ax=axes[0, 1])
axes[0, 1].set_title("2. Durée moyenne des trajets par an (en sec)", fontsize=16, fontweight='bold')
axes[0, 1].set_ylabel("Durée moyenne (secondes)")
axes[0, 1].set_xlabel("Année")

# --- G3 : Profil Horaire (Semaine vs Week-end) ---
sns.lineplot(data=pdf_horaire, x="heure_trajet", y="total_trajets", hue="type_utilisateur", style="type_jour", palette=pal_user, markers=True, dashes=False, ax=axes[1, 0], linewidth=2)
axes[1, 0].set_title("3. Profil horaire : Pendulaires (Semaine) vs Loisirs (WE)", fontsize=16, fontweight='bold')
axes[1, 0].set_xticks(range(0, 24))
axes[1, 0].set_ylabel("Nombre total de trajets")

# --- G4 : Heatmap Météo (CORRIGÉE : Fréquentation pondérée) ---
# On utilise maintenant la moyenne horaire (moyenne_trajets_par_heure)
heat_data = pdf_meteo.pivot(index="temp_arrondie", columns="vent_arrondi", values="moyenne_trajets_par_heure")
cmap_meteo = LinearSegmentedColormap.from_list("custom_meteo", ["#3498db", "#f1c40f", "#e74c3c"])
sns.heatmap(heat_data, cmap=cmap_meteo, ax=axes[1, 1], cbar_kws={'label': 'Trajets moyens par heure'})
axes[1, 1].invert_yaxis()
axes[1, 1].set_title("4. Vraie attractivité : Moyenne de trajets/heure par Météo", fontsize=16, fontweight='bold')
axes[1, 1].set_xlabel("Vitesse du vent (km/h)")
axes[1, 1].set_ylabel("Température ressentie (°C)")

# --- G5 : Distribution des durées (Violin plot) ---
print("   (Violin plot en cours, échantillonnage sécurisé...)")
pdf_sample = df.select("type_utilisateur", "duree_secondes").sample(fraction=0.05).limit(50000).toPandas()

sns.violinplot(data=pdf_sample, x="type_utilisateur", y="duree_secondes", hue="type_utilisateur", palette=pal_user, inner="quartile", ax=axes[2, 0], legend=False)
axes[2, 0].set_title("5. Distribution des durées de trajet (<2h)", fontsize=16, fontweight='bold')
axes[2, 0].set_ylabel("Durée du trajet (secondes)")

# --- G6 : Résilience à la pluie par an ---
pdf_res_total = pdf_resilience.groupby(['annee', 'type_utilisateur'])['count'].sum().reset_index(name='total_an')
pdf_res = pdf_resilience.merge(pdf_res_total, on=['annee', 'type_utilisateur'])
pdf_res['part'] = (pdf_res['count'] / pdf_res['total_an']) * 100

sns.barplot(data=pdf_res[pdf_res['is_raining'] == "Pluie"], x="annee", y="part", hue="type_utilisateur", palette=pal_user, ax=axes[2, 1])
axes[2, 1].set_title("6. Part des trajets effectués sous la pluie (>1mm/h)", fontsize=16, fontweight='bold')
axes[2, 1].set_ylabel("% du total des trajets de l'année")
axes[2, 1].set_ylim(0, 5) 

# ---------------------------------------------------------
# 5. SAUVEGARDE ET FERMETURE
# ---------------------------------------------------------
File_Out = "data/dashboard_citibike_expert_corrige.png"
print(f"💾 Dashboard sauvegardé sous '{File_Out}'...")
plt.savefig(File_Out, bbox_inches='tight')

# Affichage si possible (ex: Notebook)
plt.show()

spark.stop()
print("🎉 Analyse terminée avec succès.")
