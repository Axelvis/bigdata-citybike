import pandas as pd
from datetime import datetime
from meteostat import hourly
import os

print("🌤️  Démarrage du téléchargement de la météo horaire...")

# 1. PARAMÉTRAGES
# Identifiant officiel de la station de Central Park (New York)
station_id = '72505'
annees = range(2017, 2026) # De 2013 à 2024 inclus

liste_df = []

print("⏳ Interrogation des serveurs climatiques (par année)...")

# 2. TÉLÉCHARGEMENT SÉCURISÉ (Année par année)
for annee in annees:
    print(f"   -> Récupération de l'année {annee}...")
    date_debut = datetime(annee, 1, 1)
    date_fin = datetime(annee, 12, 31, 23, 59)
    
    # On interroge directement la station, pas le point GPS
    donnees_meteo = hourly(station_id, date_debut, date_fin)
    df_annee = donnees_meteo.fetch()
    
    # Sécurité : On vérifie que le serveur a bien répondu
    if df_annee is not None and not df_annee.empty:
        liste_df.append(df_annee)
    else:
        print(f"   ⚠️  Attention : Données manquantes pour {annee}.")

# 3. ASSEMBLAGE ET NETTOYAGE
print("\n🔗 Fusion de toutes les années...")
df_meteo = pd.concat(liste_df)

print("🧹 Nettoyage des données...")
# Meteostat place la date en index, on la transforme en colonne classique
df_meteo = df_meteo.reset_index()

# On ne garde que les colonnes utiles (Heure, Température, Pluie, Vent)
df_meteo_propre = df_meteo[['time', 'temp', 'prcp', 'wspd']].copy()

# Extraction de la date (AAAA-MM-JJ) et de l'heure (0 à 23)
df_meteo_propre['date_meteo'] = df_meteo_propre['time'].dt.date
df_meteo_propre['heure_meteo'] = df_meteo_propre['time'].dt.hour

# Traitement des valeurs manquantes (Remplacement par 0 ou propagation de la dernière température)
df_meteo_propre['prcp'] = df_meteo_propre['prcp'].fillna(0.0)
df_meteo_propre['wspd'] = df_meteo_propre['wspd'].fillna(0.0)
df_meteo_propre['temp'] = df_meteo_propre['temp'].ffill()

# 4. SAUVEGARDE EN PARQUET
dossier_final = "data/meteo_horaire_db"
os.makedirs(dossier_final, exist_ok=True)
fichier_sortie = f"{dossier_final}/historique_nyc.parquet"

print(f"💾 Sauvegarde de {len(df_meteo_propre):,} heures de météo...")
df_meteo_propre.to_parquet(fichier_sortie, index=False)

print(f"✅ TERMINÉ ! Les données horaires sont sécurisées dans : {fichier_sortie}")
