from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "Task_1.xlsx"

FEATURE_COLUMNS = [
    f"Feature {i}" for i in range(1, 11)
]
TARGET_COLUMN = "NPV"


def load_datasets(
    file_path: str | Path = DATA_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}"
        )

    df_train = pd.read_excel(
        file_path,
        sheet_name="Training dataset",
        engine="openpyxl",
    )

    df_control = pd.read_excel(
        file_path,
        sheet_name="Control dataset",
        engine="openpyxl",
    )

    return df_train, df_control


def validate_datasets(
    df_train: pd.DataFrame,
    df_control: pd.DataFrame,
) -> None:
    expected_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    for name, df in [
        ("Training dataset", df_train),
        ("Control dataset", df_control),
    ]:
        missing_columns = set(expected_columns) - set(df.columns)

        if missing_columns:
            raise ValueError(
                f"{name} is missing columns: "
                f"{sorted(missing_columns)}"
            )

        if df.empty:
            raise ValueError(f"{name} is empty.")
