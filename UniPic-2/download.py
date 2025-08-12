# -*- coding: utf-8 -*-
import os
from pathlib import Path
from huggingface_hub import snapshot_download

# --- Paramètres ---
# L'identifiant du dépôt sur Hugging Face
REPO_ID = "Skywork/UniPic2-SD3.5M-Kontext-2B"

# Le script se base sur son propre emplacement pour déterminer le répertoire racine.
SCRIPT_DIR = Path(__file__).resolve().parent

# On définit le dossier de destination comme un sous-dossier "models"
# L'opérateur / est la manière moderne de joindre des chemins avec pathlib
TARGET_DIR = SCRIPT_DIR / "models"

# --- Logique du script ---

def clone_model():
    """
    Clone un dépôt Hugging Face dans le sous-dossier "models".
    """
    print(f"Début du clonage du dépôt : {REPO_ID}")
    # On affiche le chemin de destination final
    print(f"Destination : {TARGET_DIR}")
    print("-" * 30)

    try:
        # On utilise le nouveau TARGET_DIR pour le téléchargement
        snapshot_download(
            repo_id=REPO_ID,
            local_dir=TARGET_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print("\n" + "-" * 30)
        print(f"✅ Téléchargement terminé avec succès !")
        print(f"Les fichiers du modèle se trouvent maintenant dans : {TARGET_DIR}")

    except Exception as e:
        print(f"\n❌ Une erreur est survenue pendant le téléchargement :")
        print(e)

if __name__ == "__main__":
    # Étape cruciale : créer le dossier "models" s'il n'existe pas.
    # exist_ok=True évite une erreur si le dossier existe déjà.
    print(f"Création du dossier de destination (si besoin) : {TARGET_DIR}")
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    # Lance la fonction de clonage
    clone_model()