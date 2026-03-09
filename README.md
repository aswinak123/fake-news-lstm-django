# ClarityScan Fake News Detection App

Django web app with:
- User signup/login/logout
- Keras LSTM-backed text classification endpoint
- ClarityScan is a Django web app that combines authentication with an LSTM-based text classifier. Logged-in users can submit text for classification and review their own prediction history.

## Features

- User signup, login, and logout
- Auth-protected classification page
- Keras LSTM-backed inference via stored model assets
- Per-user prediction history

## Tech stack

- Python 3.12
- Django 5
- TensorFlow / Keras
- SQLite (default)

## Project structure

- `clarityscan/` - Django project configuration
- `classifier/` - app logic (views, forms, ML service, models)
- `templates/` - HTML templates
- `ml_assets/` - model and tokenizer artifacts used at inference time

## Requirements
For Local Hosting
 - Python 3.12
 - Docker Desktop or CLI
 - DockerHub account(for remote hosting)

## Local setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. Add model artifacts (see **Model assets** below).

4. Run migrations:

```bash
python manage.py migrate
```

5. Start the app:

```bash
python manage.py runserver
```

Open `http://127.0.0.1:8000/`.

5. For remote hosting:
- Log in to Docker CLI
- Push the docker image to your DOCKERHUB:
```
docker push username/imagename:tagname
```
- Use the uploaded image to host on any host provider such as Rendr, Railway, etc...

## Model assets

Running the `lstm_build.py` file will train a new model with 'True.csv' and 'Fake.csv.' dataset in `ml_assets/models`   

OR


Place your trained model files in `ml_assets/`:

- `models/model.keras` (**required**)
- `models/tokenizer.pkl` (**required**)
- `labels.json` (optional; JSON array of class names)
- `max_len.txt` (optional; integer sequence length, default: `7000`)

If required assets are missing, prediction requests will fail with an error message shown in the UI.

## App routes

- `/signup/` create account
- `/login/` log in
- `/logged-in/` post-login landing page
- `/` classify text
- `/history/` view your own prediction history

## Running with Docker

Build and run:

```bash
docker build -t clarityscan .
docker run --rm -p 8000:8000 clarityscan
```

Then open `http://127.0.0.1:8000/`.

## Notes

- This repository includes `db.sqlite3` for convenience; you can replace it with a fresh database by deleting it and rerunning migrations.
- For production, configure a proper secret key, database, and environment-specific settings.
- The app will fail in runtime if ```bash python manage.y runserver ``` is used instead of a docker container.
