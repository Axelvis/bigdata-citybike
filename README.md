# 🚲 Citi Bike NYC - Big Data Processing Pipeline

**Projet Big Data - ENSTA Paris 2025-2026**


## 🎯 Objectif du projet
Ce projet démontre la mise en place d'un pipeline de traitement de données à l'échelle ("production-grade" et reproductible) utilisant Apache Spark. L'objectif est d'analyser un volume massif de données (environ 10 Go de logs de trajets) provenant du système de vélos en libre-service Citi Bike de New York.
Ce projet vise à analyser les flux de vélos en libre-service à New York (Citi Bike) et à modéliser la demande en croisant ces données avec l'historique météorologique de la NOAA.

## Dataset

https://s3.amazonaws.com/tripdata/index.html

## 🏗️ Architecture et Concepts Big Data
Le pipeline est divisé en deux phases distinctes pour respecter les bonnes pratiques vues en cours :

1. **Phase d'Ingestion & Stockage (CSV vers Parquet) :**

2. **Phase d'Analyse (Spark SQL & DataFrames) :**

## 🗄️ Sources des données (AWS S3 Publics)
- **Citi Bike Trips :** `s3://tripdata/`
- **Météo NOAA :** `s3://noaa-gsod-pds/`
- **Format cible :** Apache Parquet pour des performances optimales.

## ⚙️ Installation
1. Cloner le dépôt.
2. Créer un environnement virtuel.
3. Installer les dépendances : `pip install -r requirements.txt`.
4. Créer un fichier `.env` à la racine pour les variables d'environnement.
