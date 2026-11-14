from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder


NUMERICAL_FEATURES = [
    "Age",
    "Year_of_Study",
    "Sleep_Hours",
    "Study_Hours_Per_Week",
    "GPA",
    "Social_Media_Hours",
    "Screen_Time_Hours",
    "Financial_Stress",
    "Family_Support",
    "Attendance_Rate",
]

NOMINAL_FEATURES = ["Gender", "Part_Time_Job", "Previous_Mental_Health_History"]

ORDINAL_FEATURES = ["Physical_Activity", "Diet_Quality"]


def handle_outliers(X_train, X_test):

    Q1 = X_train[NUMERICAL_FEATURES].quantile(0.25)
    Q3 = X_train[NUMERICAL_FEATURES].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[NUMERICAL_FEATURES] = X_train[NUMERICAL_FEATURES].clip(
        lower=lower_bound, upper=upper_bound, axis=1
    )

    X_test[NUMERICAL_FEATURES] = X_test[NUMERICAL_FEATURES].clip(
        lower=lower_bound,
        upper=upper_bound,
        axis=1,
    )

    return X_train, X_test


def create_preprocessor():

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    nominal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    ordinal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    categories=[
                        ["Low", "Moderate", "High"],
                        ["Poor", "Average", "Good"],
                    ],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, NUMERICAL_FEATURES),
            ("nominal", nominal_pipeline, NOMINAL_FEATURES),
            ("ordinal", ordinal_pipeline, ORDINAL_FEATURES),
        ]
    )

    return preprocessor
