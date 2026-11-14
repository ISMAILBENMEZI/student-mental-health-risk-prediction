import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from src.config import CLEAN_DATA_PATH, TARGET, MODEL_PATH
from src.cleaning import clean_data
from src.preprocessing import handle_outliers
from src.training import train_models
from src.evaluation import evaluate_model

TRAIN_DATA_PATH = CLEAN_DATA_PATH.parent / "train_data.joblib"
TEST_DATA_PATH = CLEAN_DATA_PATH.parent / "test_data.joblib"


def prepare_data():

    clean_data()

    df = pd.read_csv(CLEAN_DATA_PATH)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train, X_test = handle_outliers(X_train, X_test)

    joblib.dump((X_train, y_train), TRAIN_DATA_PATH)

    joblib.dump((X_test, y_test), TEST_DATA_PATH)


def train():
    X_train, y_train = joblib.load(TRAIN_DATA_PATH)
    best_model, reuslts = train_models(X_train, y_train)


def evaluate():
    X_test, y_test = joblib.load(TEST_DATA_PATH)

    best_model = joblib.load(MODEL_PATH)

    evaluate_model(best_model, X_test, y_test)


def main():

    prepare_data()

    train()

    evaluate()


if __name__ == "__main__":
    main()
