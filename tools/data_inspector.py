import pandas as pd


class DataInspector:
    """
    Data Inspector

    Responsible for:
    - Loading datasets
    - Inspecting dataset structure
    - Providing dataset information to AI agents
    """

    def load_dataset(self, file_path):
        """
        Load a CSV dataset.
        """
        try:
            df = pd.read_csv(file_path)
            print("✅ Dataset loaded successfully!")
            return df

        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None

    def get_shape(self, df):
        """
        Returns number of rows and columns.
        """
        return {
            "rows": df.shape[0],
            "columns": df.shape[1]
        }

    def get_column_names(self, df):
        """
        Returns all column names.
        """
        return list(df.columns)

    def get_data_types(self, df):
        """
        Returns data types of all columns.
        """
        return df.dtypes.astype(str).to_dict()

    def get_missing_values(self, df):
        """
        Returns missing values in each column.
        """
        return df.isnull().sum().to_dict()

    def get_duplicate_rows(self, df):
        """
        Returns the number of duplicate rows.
        """
        return int(df.duplicated().sum())

    def get_numeric_columns(self, df):
        """
        Returns all numeric columns.
        """
        return list(df.select_dtypes(include=["number"]).columns)

    def get_categorical_columns(self, df):
        """
        Returns all categorical columns.
        """
        return list(df.select_dtypes(include=["object", "string", "category"]).columns)

    def inspect_dataset(self, df):
        """
        Returns complete dataset information.
        """
        return {
            "shape": self.get_shape(df),
            "column_names": self.get_column_names(df),
            "data_types": self.get_data_types(df),
            "missing_values": self.get_missing_values(df),
            "duplicate_rows": self.get_duplicate_rows(df),
            "numeric_columns": self.get_numeric_columns(df),
            "categorical_columns": self.get_categorical_columns(df)
        }