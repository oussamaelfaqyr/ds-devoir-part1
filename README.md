# Projet Flask + ML — Conteneurisation

## Team members:

>OUSSAMA ELFAQYR
>AYOUB MOUROU

Ce dépôt contient une application Flask qui réalise :
- classification d'images avec VGG16
- prédiction par régression (modèle sauvegardé dans `model.pkl`)

Fichiers importants :
- `Dockerfile` — image du conteneur
- `docker-compose.yml` — orchestration et volumes
- `entrypoint.sh` — script qui exécute `model.py` si `model.pkl` est absent, puis démarre `main.py` via Flask
- `images/` — dossier des images uploadées (monté comme volume)

Prérequis
- Docker
- (optionnel) docker-compose

Usage (avec docker-compose)

```bash
docker-compose up --build
```

Le service écoute sur le port `5000`. Les fichiers uploadés par l'application sont persistés sur la machine hôte dans le dossier `./images` grâce au volume défini dans `docker-compose.yml`.

Usage (sans docker-compose)

```bash
docker build -t flask-ml-app .
docker run -p 5000:5000 \
  -v %CD%/images:/app/images \
  -v %CD%/data:/app/data \
  flask-ml-app
```

Notes importantes
- Le script `entrypoint.sh` exécute `python model.py` si `model.pkl` n'existe pas dans le conteneur — utile pour générer le modèle initial.
- Les volumes montés garantissent que `images/` et `data/` persistent entre les arrêts/recréations de conteneurs.
- Sur Windows PowerShell, `%CD%` fonctionne pour monter le dossier courant ; sur Linux/macOS utilisez `$PWD`.

Développement local (sans Docker)

```bash
python -m venv .venv
.venv\Scripts\activate    # PowerShell/CMD
pip install -r requirements.txt
python model.py            # génère model.pkl
flask run --host=0.0.0.0
```

Dépannage
- Si l'image Docker échoue lors de l'installation de `tensorflow`/`keras`, utilisez une image de base différente (ex : `python:3.10-bullseye`) ou installez les dépendances système requises.
- Vérifiez les permissions sur `entrypoint.sh` (le `Dockerfile` effectue `chmod +x`).

Contact
- Si vous voulez que j'ajoute des tests automatisés, des checks d'entrée ou une image plus optimisée (Slim + wheels TensorFlow), dites‑le.
