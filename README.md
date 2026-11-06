# Customer Churn Prediction — Software Engineering for Data

## 1. Présentation du projet

Ce projet est réalisé dans le cadre du module **Software Engineering for Data (M2)**.

L'objectif est de développer une application Python de Machine Learning permettant de prédire le risque de départ (*churn*) des clients d'un opérateur télécom.

Le projet vise également à appliquer les bonnes pratiques du génie logiciel : architecture modulaire, API REST, sécurité, tests automatisés, documentation, packaging et intégration continue.

**État actuel :** première version fonctionnelle du modèle et de l'API, accompagnée de tests automatisés. La couverture de code atteint **85 %**, dépassant l'objectif minimum de 75 %. Le développement reste en cours.

## 2. Dataset

Le projet utilise le dataset **Telco Customer Churn**, diffusé par IBM.

Il contient initialement :
- **7 043 clients**
- **21 colonnes**
- Une variable cible : `Churn` (`Yes` ou `No`)

Les données comprennent notamment l'ancienneté du client, son type de contrat, les services souscrits et les montants facturés.

Après nettoyage, **7 032 observations** sont conservées.

La provenance exacte du fichier, ses conditions de réutilisation et ses limites seront documentées dans une prochaine version.

## 3. Méthodologie Machine Learning

Les étapes actuellement implémentées sont :

1. Chargement des données CSV avec Pandas.
2. Conversion de `TotalCharges` en valeurs numériques.
3. Suppression des observations présentant des valeurs manquantes dans `TotalCharges`.
4. Suppression de l'identifiant `customerID`.
5. Encodage de la variable cible : `No = 0`, `Yes = 1`.
6. Préparation des variables numériques et catégorielles avec `StandardScaler` et `OneHotEncoder`.
7. Séparation des données en 80 % pour l'entraînement et 20 % pour le test.
8. Entraînement d'un modèle de régression logistique.
9. Évaluation et sauvegarde du pipeline entraîné avec Joblib.

### Résultats préliminaires

| Indicateur | Résultat |
|---|---|
| Observations après nettoyage | 7 032 |
| Observations d'entraînement | 5 625 |
| Observations de test | 1 407 |
| Modèle | Régression logistique |
| Accuracy sur le jeu de test | **80,38 %** |

Ces résultats constituent une première référence. Des métriques complémentaires, notamment la précision, le rappel, le F1-score et la matrice de confusion, seront ajoutées.

## 4. API REST avec FastAPI

Une API REST a été développée avec FastAPI.

| Méthode | Endpoint | Fonction |
|---|---|---|
| GET | `/health` | Vérifier que l'API fonctionne |
| POST | `/predict` | Prédire le risque de départ d'un client |

L'endpoint `/predict` reçoit les caractéristiques d'un client au format JSON et renvoie une prédiction ainsi qu'une probabilité estimée de départ.

L'API utilise :
- Pydantic pour la validation des données ;
- Joblib pour charger le modèle entraîné ;
- Une clé API transmise dans l'en-tête `X-API-Key` ;
- Un fichier `.env` pour la configuration locale ;
- Swagger UI pour la documentation interactive.

### Vérifications réalisées

Les vérifications manuelles suivantes ont été effectuées :

- `GET /health` : réponse `200 OK`.
- `POST /predict` avec une clé valide : réponse `200 OK`.
- `POST /predict` sans clé : réponse `401 Unauthorized`.

Ces comportements sont désormais également vérifiés par des tests automatisés.

## 5. Tests automatisés et couverture

Les tests automatisés ont été développés avec **Pytest** et la couverture a été mesurée avec **pytest-cov**.

La suite comporte actuellement **15 tests**, tous réussis.

### Fonctionnalités testées

- **API :** fonctionnement de `/health`, refus des requêtes non autorisées et prédiction avec une clé valide.
- **Chargement des données :** type de données retourné, présence des colonnes attendues et dimensions du dataset.
- **Prétraitement :** suppression de l'identifiant, conversion numérique, encodage de la cible et traitement des valeurs manquantes.
- **Entraînement :** exécution du pipeline et vérification de la sauvegarde à l'aide d'un mock.
- **Prédiction :** vérification du type de retour et des prédictions binaires avec des modèles simulés.

### Résultats de couverture

| Module | Couverture |
|---|---:|
| `api.py` | 96 % |
| `data_loader.py` | 67 % |
| `predict.py` | 67 % |
| `preprocessing.py` | 59 % |
| `train.py` | 97 % |
| **Couverture globale** | **85 %** |

**Objectif du projet : couverture ≥ 75 %.**

**Résultat actuel : 85 % — objectif atteint.**

### Exécution des tests

Depuis la racine du projet, dans un environnement Python où les dépendances sont installées :

```bash
python -m pytest tests/ --cov=src --cov-report=term-missing
```

Dernier résultat obtenu :

```text
15 passed in 4.07s
TOTAL 135 statements, 20 missing, 85% coverage
```

La suite de tests continuera à être enrichie pendant le développement.

## 6. Structure actuelle du projet

```text
Projet_software_enginner/
├── data/
│   └── Telco-Customer-Churn.csv
├── docs/
├── models/
│   └── churn_model.joblib
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── predict.py
│   └── api.py
├── tests/
│   ├── test_api.py
│   ├── test_data_loader.py
│   ├── test_predict.py
│   ├── test_preprocessing.py
│   └── test_train.py
├── .env
├── .gitignore
└── README.md
```

Le fichier `.env` contient des informations de configuration sensibles et est exclu du dépôt Git grâce au fichier `.gitignore`.

## 7. Technologies utilisées

- Python
- Pandas
- Scikit-learn
- Joblib
- FastAPI
- Pydantic
- Uvicorn
- python-dotenv
- Pytest
- pytest-cov
- HTTPX
- unittest.mock

## 8. État d'avancement

| Fonctionnalité | Statut |
|---|---|
| Sélection et chargement du dataset | Terminé |
| Nettoyage des données | Terminé |
| Entraînement du modèle | Terminé |
| Évaluation initiale | Terminé |
| Sauvegarde du modèle | Terminé |
| Prédiction locale | Terminé |
| API FastAPI `/health` et `/predict` | Terminé |
| Authentification par clé API | Implémentée et testée |
| Tests automatisés | 15 tests réussis |
| Couverture de code ≥ 75 % | **Atteint : 85 %** |
| Logging | À faire |
| Packaging avec Poetry | À faire |
| Documentation Sphinx | À faire |
| Qualité du code avec Ruff et Black | À faire |
| Hooks pre-commit | À faire |
| CI GitHub Actions | À faire |

## 9. Prochaines étapes

Les prochaines étapes prévues sont :

1. Mettre en place Ruff et Black pour améliorer la qualité et le formatage du code.
2. Configurer les hooks pre-commit.
3. Ajouter les logs applicatifs et améliorer la gestion des erreurs.
4. Préparer le packaging du projet avec Poetry.
5. Développer la documentation Sphinx.
6. Configurer l'intégration continue avec GitHub Actions.
7. Compléter l'évaluation du modèle et enrichir les tests.



**Projet en cours de développement — M2 Software Engineering for Data.**