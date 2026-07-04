from typing import Dict, List, Any

import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler


class Cleaner:
    """
    Cleaner Tool

    This class performs all dataset preprocessing operations.

    Responsibilities
    ----------------
    - Validate inputs
    - Drop unnecessary columns
    - Remove duplicate rows
    - Fill missing values
    - Encode categorical variables
    - Scale numerical variables

    Every preprocessing step is recorded inside the cleaning log.
    """

    def __init__(self) -> None:
        self.cleaning_log: List[str] = []

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

    def _validate_schema(self, schema: Dict[str, Any]) -> None:
        if not isinstance(schema, dict):
            raise TypeError("Schema must be a Python dictionary.")

    def _log(self, message: str) -> None:
        self.cleaning_log.append(message)

    def get_cleaning_log(self) -> List[str]:
        return self.cleaning_log

    # ---------------- DROP COLUMNS ----------------
    def drop_columns(self, df: pd.DataFrame, schema: Dict[str, Any]) -> pd.DataFrame:
        self._validate_dataframe(df)
        self._validate_schema(schema)

        columns_to_drop = schema.get("drop_columns", [])

        if not columns_to_drop:
            self._log("No columns were dropped.")
            return df

        existing_columns = [c for c in columns_to_drop if c in df.columns]

        if existing_columns:
            df = df.drop(columns=existing_columns)
            self._log(f"Dropped columns: {', '.join(existing_columns)}")
        else:
            self._log("Schema suggested dropping columns, but none existed.")

        return df

    # ---------------- REMOVE DUPLICATES ----------------
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        self._validate_dataframe(df)

        dup_count = int(df.duplicated().sum())

        if dup_count == 0:
            self._log("No duplicate rows found.")
            return df

        df = df.drop_duplicates()
        self._log(f"Removed {dup_count} duplicate rows.")

        return df

    # ---------------- MISSING VALUES ----------------
    def fill_missing_values(self, df: pd.DataFrame, schema: Dict[str, Any]) -> pd.DataFrame:
        self._validate_dataframe(df)
        self._validate_schema(schema)

        strategies = schema.get("missing_value_strategy", {})

        if not strategies:
            self._log("No missing value strategy found.")
            return df

        for col, strategy in strategies.items():

            if col not in df.columns:
                continue

            if strategy == "drop":
                continue

            if strategy == "mean":
                imputer = SimpleImputer(strategy="mean")
            elif strategy == "median":
                imputer = SimpleImputer(strategy="median")
            elif strategy == "mode":
                imputer = SimpleImputer(strategy="most_frequent")
            else:
                self._log(f"Unknown strategy {strategy} for {col}")
                continue

            df[[col]] = imputer.fit_transform(df[[col]])

            self._log(f"Filled {col} using {strategy}")

        return df

    # ---------------- ENCODING ----------------
    def encode_categorical_columns(self, df: pd.DataFrame, schema: Dict[str, Any]) -> pd.DataFrame:
        self._validate_dataframe(df)
        self._validate_schema(schema)

        cols = schema.get("encoding_required", [])

        if not cols:
            self._log("No encoding required.")
            return df

        for col in cols:

            if col not in df.columns:
                continue

            unique_vals = df[col].nunique()

            # Binary encoding
            if unique_vals == 2:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self._log(f"Label encoded {col}")

            # One-hot encoding
            else:
                dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                self._log(f"One-hot encoded {col}")

        return df

    # ---------------- SCALING ----------------
    def scale_numeric_columns(self, df: pd.DataFrame, schema: Dict[str, Any]) -> pd.DataFrame:
        self._validate_dataframe(df)
        self._validate_schema(schema)

        cols = schema.get("scaling_required", [])

        if not cols:
            self._log("No scaling required.")
            return df

        scaler = StandardScaler()

        valid_cols = [c for c in cols if c in df.columns]

        if not valid_cols:
            self._log("No valid numeric columns to scale.")
            return df

        df[valid_cols] = scaler.fit_transform(df[valid_cols])

        self._log(f"Scaled columns: {', '.join(valid_cols)}")

        return df

    # ---------------- FULL PIPELINE ----------------
    def clean_dataset(self, df: pd.DataFrame, schema: Dict[str, Any]) -> pd.DataFrame:
        self._validate_dataframe(df)
        self._validate_schema(schema)

        self._log("Starting full cleaning pipeline")

        df = self.drop_columns(df, schema)
        df = self.remove_duplicates(df)
        df = self.fill_missing_values(df, schema)
        df = self.encode_categorical_columns(df, schema)
        df = self.scale_numeric_columns(df, schema)

        self._log("Cleaning pipeline completed")

        return df