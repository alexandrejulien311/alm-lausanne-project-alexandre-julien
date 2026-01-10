import pandas as pd
from pathlib import Path
from data.dataset_raw import DATASET_RAW_PATH
from data.dataset_clean import DATASET_CLEAN_PATH

# Data form Kaggle : https://www.kaggle.com/datasets/dsfelix/employees-payroll-in-los-angeles
# Charastics of L.A. civil servants

file_path = Path() / DATASET_RAW_PATH / "payroll_LA.csv"
file_destination = Path() / DATASET_CLEAN_PATH / "payroll_LA_clean.csv"

df = pd.read_csv(file_path)

df = df[df["JOB_STATUS"] == "ACTIVE"]
df = df[df["REGULAR_PAY"] > 0]
df = df.drop(
    columns=[
        "PAY_YEAR",
        "DEPARTMENT_NO",
        "JOB_CLASS_PGRADE",
        "JOB_STATUS",
        "MOU",
        "REGULAR_PAY",
        "OVERTIME_PAY",
        "ALL_OTHER_PAY",
        "RECORD_NBR",
        "DEPARTMENT_TITLE",
        "JOB_TITLE",
    ]
)
df.to_csv(file_destination)
