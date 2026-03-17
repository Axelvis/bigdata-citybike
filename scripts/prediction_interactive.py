import joblib
import pandas as pd
import sys

# ---------------------------------------------------------
# 1. INITIALISATION ET CHARGEMENT DU MODÈLE
# ---------------------------------------------------------
print("🧠 Chargement de l'Intelligence Artificielle en cours...")
try:
    modele = joblib.load("data/modele_citibike_tpot.pkl")
    print("✅ Modèle chargé et prêt à l'emploi !\n")
except FileNotFoundError:
    print("❌ Erreur : Le fichier 'modele_citibike_tpot.pkl' est introuvable.")
    print("Avez-vous bien exécuté la sauvegarde à la fin du script de modélisation ?")
    sys.exit()

print("=" * 60)
print(" 🚲 BIENVENUE DANS LE SIMULATEUR DE DEMANDE CITI BIKE 🚲 ")
print("=" * 60)
print("Appuyez sur 'q' et Entrée à tout moment pour quitter le programme.\n")

# ---------------------------------------------------------
# 2. BOUCLE INTERACTIVE
# ---------------------------------------------------------
while True:
    print("-" * 60)
    print("Veuillez entrer les prévisions pour votre simulation :")
    
    # Récupération des inputs de l'utilisateur
    temp_input = input("🌡️  Température prévue (en °F, ex: 75) : ")
    if temp_input.lower() == 'q': break
        
    precip_input = input("🌧️  Précipitations (en pouces, ex: 0 pour sec, 0.5 pour pluie) : ")
    if precip_input.lower() == 'q': break
        
    mois_input = input("📅 Mois de l'année (1 à 12, ex: 7 pour Juillet) : ")
    if mois_input.lower() == 'q': break
        
    # PySpark utilise cette norme : 1=Dim, 2=Lun, 3=Mar, 4=Mer, 5=Jeu, 6=Ven, 7=Sam
    jour_input = input("📆 Jour (1=Dim, 2=Lun, 3=Mar, 4=Mer, 5=Jeu, 6=Ven, 7=Sam) : ")
    if jour_input.lower() == 'q': break

    # ---------------------------------------------------------
    # 3. TRAITEMENT ET PRÉDICTION
    # ---------------------------------------------------------
    try:
        # Conversion des textes tapés en nombres compréhensibles par l'IA
        temp = float(temp_input)
        precip = float(precip_input)
        mois = int(mois_input)
        jour = int(jour_input)
        
        # Création du format attendu par le modèle (DataFrame Pandas)
        donnees_simulation = pd.DataFrame({
            "temperature_f": [temp],
            "precipitations_pouces": [precip],
            "mois": [mois],
            "jour_semaine": [jour]
        })
        
        # L'IA fait sa prédiction
        prediction = modele.predict(donnees_simulation)
        
        # On arrondit pour avoir un nombre entier de vélos (on ne loue pas un demi-vélo !)
        nb_velos = int(prediction[0])
        
        print("\n✨ " + "=" * 45 + " ✨")
        print(f"🔮 RÉSULTAT DE L'IA : Environ {nb_velos:,} locations attendues.")
        print("✨ " + "=" * 45 + " ✨\n")
        
    except ValueError:
        # Sécurité au cas où l'utilisateur tape des lettres au lieu de chiffres
        print("\n⚠️  Oups ! Veuillez entrer uniquement des nombres valides avec un point (ex: 75.5). Réessayons.\n")

print("\n👋 Merci d'avoir utilisé le simulateur Citi Bike. À bientôt !")
