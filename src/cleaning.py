import pandas as pd
from src.config import RAW_DATA_PATH, CLEAN_DATA_PATH


def clean_data():
    df = pd.read_csv(RAW_DATA_PATH)

    invalid_rows = df["Social_Media_Hours"] > df["Screen_Time_Hours"]

    df.loc[invalid_rows, "Social_Media_Hours"] = df.loc[
        invalid_rows, "Screen_Time_Hours"
    ]

    df = df.drop(columns=["Student_ID"])

    df.to_csv(CLEAN_DATA_PATH , index=False)

    
