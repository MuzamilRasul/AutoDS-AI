from pprint import pprint

from tools.data_inspector import DataInspector
from agents.schema_agent import SchemaAgent


def main():

    # Load dataset
    inspector = DataInspector()

    df = inspector.load_dataset("data/raw/train.csv")

    # Inspect dataset
    dataset_info = inspector.inspect_dataset(df)

    # Create Schema Agent
    agent = SchemaAgent()

    # Analyze schema
    result = agent.analyze_schema(dataset_info)

    print("\n" + "=" * 60)
    print("SCHEMA AGENT OUTPUT")
    print("=" * 60)

    pprint(result)


if __name__ == "__main__":
    main()