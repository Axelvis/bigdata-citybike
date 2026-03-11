from pyspark.sql import SparkSession
from pyspark.sql.functions import col, desc

def main():
    spark = SparkSession.builder \
        .appName("CitiBike-Analytics") \
        .master("local[*]") \
        .getOrCreate()

    print("Lecture des données Parquet...")
    # On lit les données optimisées (Parquet ne lira que les colonnes nécessaires) 
    df = spark.read.parquet("data/processed/citibike_trips.parquet")

    # --- ANALYSE 1 : Les 10 stations de départ les plus populaires ---
    print("\n--- Top 10 des stations de départ ---")
    top_stations = df.groupBy("start_station_name") \
                     .count() \
                     .orderBy(desc("count"))
    
    # L'action .show() déclenche le calcul [cite: 90]
    top_stations.show(10, truncate=False)

    # --- ANALYSE 2 : Types de vélos les plus utilisés ---
    print("\n--- Répartition par type de vélo ---")
    bike_types = df.groupBy("rideable_type") \
                   .count() \
                   .orderBy(desc("count"))
                   
    bike_types.show()

    spark.stop()

if __name__ == "__main__":
    main()
