# 🚲 Citi Bike NYC - Big Data Processing Pipeline

**Projet Big Data - ENSTA Paris 2025-2026**


## 🎯 Objectif du projet
Ce projet démontre la mise en place d'un pipeline de traitement de données à l'échelle ("production-grade" et reproductible) utilisant Apache Spark. L'objectif est d'analyser un volume massif de données (environ 10 Go de logs de trajets) provenant du système de vélos en libre-service Citi Bike de New York.

## Dataset

https://s3.amazonaws.com/tripdata/index.html

## 🏗️ Architecture et Concepts Big Data
Le pipeline est divisé en deux phases distinctes pour respecter les bonnes pratiques vues en cours :

1. **Phase d'Ingestion & Stockage (CSV vers Parquet) :**

2. **Phase d'Analyse (Spark SQL & DataFrames) :**
  
## Structure du dépôt
```text
citibike-spark-pipeline/
├── data/
│   ├── raw/             <- (Ignoré par Git) Fichiers CSV bruts de Citi Bike
│   └── processed/       <- (Ignoré par Git) Données converties au format Parquet
├── src/
│   ├── 01_ingestion.py  <- Job Spark d'ingestion et conversion
│   └── 02_analytics.py  <- Job Spark d'analyse (DataFrames)
├── .gitignore           <- Configuration Git (exclut le dossier data/)
├── requirements.txt     <- Dépendances Python
└── README.md            <- Documentation du projet
