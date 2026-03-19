"""Run the full pipeline: generate data, train, and evaluate."""

from .generate_data import generate_dataset, save_dataset
from .train import train, TrainConfig

if __name__ == "__main__":
    # Generate data
    records = generate_dataset(n_samples=20000)
    save_dataset(records)

    # Train BiLSTM
    print("\n" + "=" * 55)
    print("Training BiLSTM")
    print("=" * 55)
    bilstm = train(TrainConfig(model_type="bilstm"))

    # Train CNN
    print("\n" + "=" * 55)
    print("Training CNN")
    print("=" * 55)
    cnn = train(TrainConfig(model_type="cnn"))
