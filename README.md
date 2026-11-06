# Customer Churn Prediction — Software Engineering for Data

## 1. Présentation du projet

Ce projet est réalisé dans le cadre du module **Software Engineering for Data (M2)**.

L'objectif est de développer une application Python de Machine Learning permettant de prédire le risque de départ (*churn*) des clients d'un opérateur télécom.

Le projet vise également à appliquer les bonnes pratiques du génie logiciel : architecture modulaire, API REST, sécurité, tests automatisés, documentation, packaging et intégration continue.

**État actuel :** première version fonctionnelle du modèle et de l'API. Le développement est toujours en cours.

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

### Tests manuels déjà effectués

- `GET /health` : réponse `200 OK`.
- `POST /predict` avec une clé valide : réponse `200 OK`.
- `POST /predict` sans clé : réponse `401 Unauthorized`.

Ces vérifications sont manuelles. Les tests automatisés restent à développer.

## 5. Structure actuelle du projet

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
│   └── test_data_loader.py
├── .env
├── .gitignore
└── README.md
```

Le fichier `.env` contient des informations de configuration sensibles et ne doit pas être publié sur GitHub.

## 6. Technologies utilisées

- Python
- Pandas
- Scikit-learn
- Joblib
- FastAPI
- Pydantic
- Uvicorn
- python-dotenv

## 7. État d'avancement

| Fonctionnalité | Statut |
|---|---|
| Sélection et chargement du dataset | Terminé |
| Nettoyage des données | Terminé |
| Entraînement du modèle | Terminé |
| Évaluation initiale | Terminé |
| Sauvegarde du modèle | Terminé |
| Prédiction locale | Terminé |
| API FastAPI `/health` et `/predict` | Terminé |
| Authentification par clé API | Terminé — tests manuels |
| Tests automatisés et couverture ≥ 75 % | À faire |
| Logging | À faire |
| Packaging avec Poetry | À faire |
| Documentation Sphinx | À faire |
| Qualité du code et pre-commit | À faire |
| CI GitHub Actions | À faire |

## 8. Prochaines étapes

Les prochaines étapes prévues sont :

1. Développer les tests unitaires et d'intégration avec Pytest.
2. Mesurer la couverture de code avec pytest-cov.
3. Améliorer la validation des données et la gestion des erreurs.
4. Ajouter les logs applicatifs.
5. Préparer le packaging avec Poetry.
6. Ajouter la documentation Sphinx.
7. Configurer le linting et la CI GitHub Actions.

## 9. Utilisation de l'intelligence artificielle

ChatGPT a été utilisé comme outil d'assistance pour expliquer les concepts, proposer des exemples de code, accompagner le débogage et aider à structurer la documentation.

Les fonctionnalités ont été exécutées et vérifiées progressivement dans l'environnement de développement. Les tests automatisés et la validation finale restent à réaliser.

---

**Projet en cours de développement — M2 Software Engineering for Data.**