import pandas as pd
import joblib
from pyspark.sql import SparkSession
from pyspark.sql.functions import count, month, dayofweek, col
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

if __name__ == '__main__':
    print("Demarrage de la session PySpark pour l'analyse par station...")
    spark = SparkSession.builder \
        .appName("CitiBike_Station_Prediction") \
        .getOrCreate()

    # ---------------------------------------------------------
    # 1. TRAITEMENT DISTRIBUE (Spark)
    # ---------------------------------------------------------
    print("Lecture de la base de donnees unifiee...")
    df_final = spark.read.parquet("data/dataset_final_modelisation")

    # --- Standardisation de la colonne de la station ---
    # On s'adapte aux changements de noms de colonnes de Citi Bike au fil des ans
    col_station = "start_station_name" if "start_station_name" in df_final.columns else "start station name"
    df_final = df_final.withColumnRenamed(col_station, "station_depart")

    # Nettoyage des lignes sans station de depart
    df_clean = df_final.dropna(subset=["station_depart"])

    # Extraction des variables temporelles
    df_features = df_clean.withColumn("mois", month("date_trajet")) \
                        .withColumn("jour_semaine", dayofweek("date_trajet"))

    # --- Identification des 10 stations les plus importantes ---
    print("Recherche des 10 stations les plus frequentees...")
    top_stations_df = df_features.groupBy("station_depart") \
        .count() \
        .orderBy(col("count").desc()) \
        .limit(10)

    # Extraction des noms des stations sous forme de liste Python
    top_stations_list = [row['station_depart'] for row in top_stations_df.collect()]
    print(f"Stations selectionnees : {top_stations_list}")

    # --- Filtrage et Agregation ---
    print("Calcul des departs quotidiens pour ces stations...")
    df_top_stations = df_features.filter(col("station_depart").isin(top_stations_list))

    df_station_daily = df_top_stations.groupBy(
        "date_trajet", "station_depart", "temperature_f", "precipitations_pouces", "mois", "jour_semaine"
    ).agg(count("*").alias("departs_quotidiens"))

    # ---------------------------------------------------------
    # 2. BASCULE VERS PANDAS ET PREPARATION ML
    # ---------------------------------------------------------
    print("Conversion en Pandas pour le Machine Learning...")
    df_ml = df_station_daily.toPandas().dropna()

    # Encodage de la variable categorielle (Nom de la station)
    # L'algorithme a besoin de nombres, on transforme les noms en colonnes binaires (0 ou 1)
    df_ml = pd.get_dummies(df_ml, columns=['station_depart'], drop_first=True)

    # Separation des caracteristiques (X) et de la cible (y)
    X = df_ml.drop(columns=['date_trajet', 'departs_quotidiens'])
    y = df_ml['departs_quotidiens']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ---------------------------------------------------------
    # 3. MACHINE LEARNING AVEC TPOT
    # ---------------------------------------------------------
    print("\nLancement de TPOT pour la prediction par station...")
    print("Attention : Cela peut prendre quelques minutes...")

    tpot = TPOTRegressor(
        generations=5, 
        population_size=20, 
        verbose=2, 
        random_state=42, 
        n_jobs=6
    )

    tpot.fit(X_train, y_train)

    # ---------------------------------------------------------
    # 4. EVALUATION ET EXPORT
    # ---------------------------------------------------------

    print("\n✅ Entrainement termine !")

    predictions = tpot.predict(X_test)
    score_r2 = r2_score(y_test, predictions)

    print("-" * 50)
    print(f"🎯 Score R2 du modele par station : {score_r2:.2f}")
    print("-" * 50)

    # SAUVEGARDE CORRIGÉE POUR NE PAS CRASHER À LA FIN
    fichier_modele = "data/modele_stations_tpot.pkl"
    joblib.dump(tpot.fitted_pipeline_, fichier_modele)
    
    print(f"💾 Modele complet sauvegarde dans : {fichier_modele}")

    spark.stop()
