#!/bin/sh
set -e

# If model.pkl doesn't exist, run model.py to generate it
if [ ! -f model.pkl ]; then
  echo "model.pkl not found — running model.py to generate it..."
  python model.py
else
  echo "model.pkl exists — skipping model generation."
fi

# Start the Flask app
exec flask run --host=0.0.0.0
