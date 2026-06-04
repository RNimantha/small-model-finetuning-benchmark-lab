from datasets import load_dataset


def main():
    dataset_name = "paraloq/json_data_extraction"

    dataset = load_dataset(dataset_name)

    print(dataset)
    print("\nAvailable splits:")
    print(dataset.keys())

    split_name = list(dataset.keys())[0]
    print(f"\nFirst example from split: {split_name}")
    print(dataset[split_name][0])

    print("\nColumns:")
    print(dataset[split_name].column_names)


if __name__ == "__main__":
    main()