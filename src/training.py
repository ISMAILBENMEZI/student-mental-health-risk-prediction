import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from src.preprocessing import create_preprocessor
from src.config import MODEL_PATH


def train_models(X_train, y_train):
    models = {
        "Logistic Regression": {
            "model": LogisticRegression(),
            "params": {"classifier__C": [0.01, 0.1, 1, 10]},
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "classifier__n_estimators": [100, 200, 300],
                "classifier__max_depth": [None, 10, 20],
                "classifier__min_samples_split": [2, 5],
            },
        },
        "KNN": {
            "model": KNeighborsClassifier(),
            "params": {
                "classifier__n_neighbors": [3, 5, 7, 9],
                "classifier__weights": ["uniform", "distance"],
            },
        },
    }

    results = []
    best_model = None
    best_score = 0

    for model_name, model_info in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", create_preprocessor()),
                ("classifier", model_info["model"]),
            ]
        )

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=model_info["params"],
            cv=5,
            scoring="f1_weighted",
            n_jobs=-1,
        )

        grid_search.fit(X_train, y_train)

        results.append(
            {
                "Model": model_name,
                "Best CV F1": grid_search.best_score_,
                "Best Parameters": grid_search.best_params_,
            }
        )

        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_model = grid_search.best_estimator_

    results_df = pd.DataFrame(results)

    joblib.dump(best_model , MODEL_PATH)

    return best_model , results_df

