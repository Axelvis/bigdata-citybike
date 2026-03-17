import pandas as pd
from datetime import datetime
from meteostat import Point, hourly
import os

print("🌤️  Démarrage du téléchargement de la météo horaire...")

# 1. PARAMÉTRAGES
# Coordonnées géographiques de Central Park, New York
ny_central_park = Point(40.7831, -73.9712)

# Période globale du projet (De l'ouverture de Citi Bike à fin 2024)
date_debut = datetime(2013, 1, 1)
date_fin = datetime(2024, 12, 31, 23, 59)

# 2. TÉLÉCHARGEMENT DES DONNÉES
print(f"⏳ Interrogation des serveurs climatiques de {date_debut.year} à {date_fin.year}...")
# On récupère les données horaires
donnees_meteo = hourly(ny_central_park, date_debut, date_fin)
df_meteo = donnees_meteo.fetch()

# 3. NETTOYAGE ET PRÉPARATION
print("🧹 Nettoyage des données...")

# Meteostat met la date/heure en index, on la transforme en vraie colonne
df_meteo = df_meteo.reset_index()

# On ne garde que les colonnes qui nous intéressent vraiment
# time: Date et Heure
# temp: Température en °Celsius !
# prcp: Précipitations en millimètres
# wspd: Vitesse du vent en km/h
df_meteo_propre = df_meteo[['time', 'temp', 'prcp', 'wspd']].copy()

# On sépare la date et l'heure dans deux colonnes distinctes pour faciliter 
# la future jointure avec les trajets de vélos
df_meteo_propre['date_meteo'] = df_meteo_propre['time'].dt.date
df_meteo_propre['heure_meteo'] = df_meteo_propre['time'].dt.hour

# Remplacer les éventuelles valeurs manquantes (NaN) par des zéros pour la pluie et le vent
df_meteo_propre['prcp'] = df_meteo_propre['prcp'].fillna(0.0)
df_meteo_propre['wspd'] = df_meteo_propre['wspd'].fillna(0.0)
# Pour la température, on propage la température de l'heure précédente s'il y a un trou
df_meteo_propre['temp'] = df_meteo_propre['temp'].ffill()

# 4. SAUVEGARDE EN PARQUET
dossier_final = "data/meteo_horaire_db"
os.makedirs(dossier_final, exist_ok=True)
fichier_sortie = f"{dossier_final}/historique_nyc.parquet"

print(f"💾 Sauvegarde de {len(df_meteo_propre):,} heures de météo...")
df_meteo_propre.to_parquet(fichier_sortie, index=False)

print(f"✅ TERMINÉ ! Les données sont sécurisées dans : {fichier_sortie}")
