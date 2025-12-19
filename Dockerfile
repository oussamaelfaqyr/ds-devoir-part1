# Utiliser une image de base Python 3.10
FROM python:3.10-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers requirements et installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier tous les fichiers de l'application
COPY . .

# Copier le script d'entrée et le rendre exécutable
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Exposer le port Flask (par défaut 5000)
EXPOSE 5000

# Définir les variables d'environnement
ENV FLASK_APP=main.py
ENV FLASK_ENV=production

# Utiliser le script d'entrée pour lancer le pipeline (model.py puis Flask)
CMD ["/app/entrypoint.sh"]
