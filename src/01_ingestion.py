from pyspark.sql import SparkSession

def main():
    # 1. Initialiser la session Spark (mode local)
    spark = SparkSession.builder \
        .appName("CitiBike-Ingestion") \
        .master("local[*]") \
        .getOrCreate()

    print("Lecture des fichiers CSV bruts...")
    
    # 2. Charger les données CSV
    # On suppose que vous exécutez le script depuis la racine du projet
    df_raw = spark.read.csv(
        "data/raw/*.csv", 
        header=True, 
        inferSchema=True
    )
    
    print(f"Nombre de lignes chargées : {df_raw.count()}")

    # 3. Écrire en format Parquet
    # Le format Parquet offre une meilleure compression et gère l'évolution du schéma[cite: 375, 376].
    print("Conversion et écriture au format Parquet...")
    df_raw.write \
        .mode("overwrite") \
        .parquet("data/processed/citibike_trips.parquet")
        
    print("Ingestion terminée avec succès !")
    spark.stop()

if __name__ == "__main__":
    main()
