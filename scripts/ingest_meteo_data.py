import pandas as pd
from datetime import datetime
from meteostat import hourly, config
import os

print("🌤️  Démarrage du téléchargement multi-stations...")
config.block_large_requests = False

# Les 4 piliers météorologiques de la région de NYC
stations_nyc = {
    'Central_Park': '72505',
    'LaGuardia': '72503',
    'JFK': '74486',
    'Newark': '72502'
}

annees = range(2017, 2026)
liste_globale_df = []

# Double boucle : Par station, puis par année
for nom_station, station_id in stations_nyc.items():
    print(f"\n📍 Traitement de la station : {nom_station.replace('_', ' ')}")
    
    for annee in annees:
        date_debut = datetime(annee, 1, 1)
        date_fin = datetime(annee, 12, 31, 23, 59)
        
        donnees_meteo = hourly(station_id, date_debut, date_fin)
        df_annee = donnees_meteo.fetch()
        
        if df_annee is not None and not df_annee.empty:
            df_annee = df_annee.reset_index()
            # On ajoute le nom de la station pour pouvoir les différencier plus tard !
            df_annee['nom_station_meteo'] = nom_station 
            liste_globale_df.append(df_annee)
        else:
            print(f"   ⚠️ Données manquantes en {annee}")

print("\n🔗 Fusion des stations et nettoyage...")
df_meteo = pd.concat(liste_globale_df)

df_meteo_propre = df_meteo[['time', 'nom_station_meteo', 'temp', 'prcp', 'wspd']].copy()

df_meteo_propre['date_meteo'] = df_meteo_propre['time'].dt.date
df_meteo_propre['heure_meteo'] = df_meteo_propre['time'].dt.hour

df_meteo_propre['prcp'] = df_meteo_propre['prcp'].fillna(0.0)
df_meteo_propre['wspd'] = df_meteo_propre['wspd'].fillna(0.0)
df_meteo_propre['temp'] = df_meteo_propre['temp'].ffill()

dossier_final = "data/meteo_horaire_multi_db"
os.makedirs(dossier_final, exist_ok=True)
fichier_sortie = f"{dossier_final}/historique_nyc_complet.parquet"

df_meteo_propre.to_parquet(fichier_sortie, index=False)
print(f"✅ TERMINÉ ! {len(df_meteo_propre):,} lignes sauvegardées dans {fichier_sortie}")
