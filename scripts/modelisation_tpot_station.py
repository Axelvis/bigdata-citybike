import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import count, month, dayofweek, col
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

print("Demarrage de la session PySpark pour l'analyse par station...")
spark = SparkSession.builder \
    .appName("CitiBike_Station_Prediction") \
    .getOrCreate()

# ---------------------------------------------------------
# 1. TRAITEMENT DISTRIBUE (Spark)
# ---------------------------------------------------------
print("Lecture de la base de donnees unifiee...")
# Lecture optimisee grace au format Parquet en colonnes[cite: 374].
df_final = spark.read.parquet("../data/dataset_final_modelisation")

# Nettoyage des lignes sans station de depart
df_clean = df_final.dropna(subset=["start_station_name"])

# Extraction des variables temporelles
df_features = df_clean.withColumn("mois", month("date_trajet")) \
                      .withColumn("jour_semaine", dayofweek("date_trajet"))

# --- Identification des 10 stations les plus importantes ---
print("Recherche des 10 stations les plus frequentees...")
top_stations_df = df_features.groupBy("start_station_name") \
    .count() \
    .orderBy(col("count").desc()) \
    .limit(10)

# Extraction des noms des stations sous forme de liste Python
top_stations_list = [row['start_station_name'] for row in top_stations_df.collect()]
print(f"Stations selectionnees : {top_stations_list}")

# --- Filtrage et Agregation ---
print("Calcul des departs quotidiens pour ces stations...")
# On filtre le dataset massif pour ne garder que nos 10 stations
df_top_stations = df_features.filter(col("start_station_name").isin(top_stations_list))

# On groupe par jour, meteo et nom de la station
# Le moteur Catalyst optimise ce plan logique avant execution[cite: 155].
df_station_daily = df_top_stations.groupBy(
    "date_trajet", "start_station_name", "temperature_f", "precipitations_pouces", "mois", "jour_semaine"
).agg(count("*").alias("departs_quotidiens"))

# ---------------------------------------------------------
# 2. BASCULE VERS PANDAS ET PREPARATION ML
# ---------------------------------------------------------
print("Conversion en Pandas pour le Machine Learning...")
# L'action d'export declenche le calcul distribue
df_ml = df_station_daily.toPandas().dropna()

# Encodage de la variable categorielle (Nom de la station)
# L'algorithme a besoin de nombres, on transforme les noms en colonnes binaires (0 ou 1)
df_ml = pd.get_dummies(df_ml, columns=['start_station_name'], drop_first=True)

# Separation des caracteristiques (X) et de la cible (y)
# On exclut la date qui n'est pas une variable numerique predictive
X = df_ml.drop(columns=['date_trajet', 'departs_quotidiens'])
y = df_ml['departs_quotidiens']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# 3. MACHINE LEARNING AVEC TPOT
# ---------------------------------------------------------
print("\nLancement de TPOT pour la prediction par station...")

tpot = TPOTRegressor(
    generations=5, 
    population_size=20, 
    verbose=2, 
    random_state=42, 
    n_jobs=1
)

tpot.fit(X_train, y_train)

# ---------------------------------------------------------
# 4. EVALUATION ET EXPORT
# ---------------------------------------------------------
print("\nEntrainement termine !")

predictions = tpot.predict(X_test)
score_r2 = r2_score(y_test, predictions)

print("-" * 50)
print(f"Score R2 du modele de prediction par station : {score_r2:.2f}")
print("-" * 50)

fichier_export = "modele_stations_pipeline.py"
tpot.export(fichier_export)
print(f"Code source du meilleur modele sauvegarde dans : {fichier_export}")

spark.stop()
