import argparse

def add_common_training_args(parser: argparse.ArgumentParser) -> None:
    """
    Adds basic arguments needed for a training script.
    """
    parser.add_argument("--input_train_path", type=str, required=True, help="Input Path to training data.")
    parser.add_argument("--input_validation_path", type=str, required=True, help="Validation data input path")
    parser.add_argument("--output_directory", type=str, required=True, help="Directory path for all output artifacts from training job")
    parser.add_argument("--logging_directory", type=str, required=True, help="Base directory where all logging files will be stored, including tensorboard")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs to run")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")

    return
