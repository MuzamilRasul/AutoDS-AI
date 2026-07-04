from tools.data_inspector import DataInspector

# Create Data Inspector
inspector = DataInspector()

# Load Dataset
df = inspector.load_dataset("data/raw/train.csv")

if df is not None:

    dataset_info = inspector.inspect_dataset(df)

    print("\n========== DATASET INSPECTION ==========\n")

    for key, value in dataset_info.items():
        print(f"{key.upper()}:")
        print(value)
        print("-" * 50)