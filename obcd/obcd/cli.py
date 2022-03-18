import argparse
from argparse import ArgumentParser
from enum import Enum
from typing import Optional
from typing import Union

from obcd.extract import FeatureExtractor


class COMMANDS(Enum):
    EXTRACT = "extract"
    TRAIN = "train"
    PREDICT = "predict"


def setup_feature_extractor(
    subparsers: argparse._SubParsersAction,
) -> argparse._SubParsersAction:
    parser_extract = subparsers.add_parser(
        COMMANDS.EXTRACT.value,
        help="Feature extractor to create training data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_extract.add_argument(
        "infile",
        type=str,
        help="Path to L8 Metadata '*_MTL.txt`",
        default=argparse.SUPPRESS,
    )
    parser_extract.add_argument(
        "--csv", help="Save to csv", default=False, action="store_true"
    )
    parser_extract.add_argument(
        "--sqlite",
        help="Save to sqlite database",
        default=False,
        action="store_true",
    )
    parser_extract.add_argument(
        "--db_name",
        type=str,
        help="Sqlite database name",
        default="OBCD-features.db",
    )
    parser_extract.add_argument(
        "--biome_path",
        type=Optional[str],
        help="Path to L8 Biome '*_fixedmask.img`. Auto-detected if not provided",
        default=None,
    )
    parser_extract.add_argument(
        "--labels_path",
        type=Optional[str],
        help="Path to saved quickshift labels",
        default=None,
    )
    parser_extract.add_argument(
        "--temp_dir",
        type=Optional[str],
        help="Path to L8 Biome '*_fixedmask.img`",
        default=None,
    )
    parser_extract.add_argument(
        "--save_labels",
        type=bool,
        help="Bool to save quickshift labels result",
        default=False,
    )
    parser_extract.add_argument(
        "--kernel_size", type=Union[float, int], help="Quickshift attribute", default=1
    )
    parser_extract.add_argument(
        "--max_distance", type=Union[float, int], help="Quickshift attribute", default=7
    )
    parser_extract.add_argument(
        "--ratio", type=float, help="Quickshift attribute", default=0.75
    )
    parser_extract.add_argument(
        "--convert2lab", type=bool, help="Quickshift attribute", default=False
    )

    return subparsers


def setup_train(
    subparsers: argparse._SubParsersAction,
) -> argparse._SubParsersAction:

    parser_train = subparsers.add_parser(
        COMMANDS.TRAIN.value,
        help="Train model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    return subparsers


def setup_predict(
    subparsers: argparse._SubParsersAction,
) -> argparse._SubParsersAction:

    parser_predict = subparsers.add_parser(
        COMMANDS.PREDICT.value,
        help="Predict using model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    return subparsers


def app() -> None:
    parser = ArgumentParser(prog="Object Based Cloud Detection")

    subparsers = parser.add_subparsers(help="", dest="command")
    # title="subcommands",  description="valid subcommands",

    ##
    # Feature Extractor Command
    ##
    subparsers = setup_feature_extractor(subparsers)

    ##
    # Training Command
    ##
    subparsers = setup_train(subparsers)

    ##
    # Predict Command
    ##
    subparsers = setup_predict(subparsers)

    args = parser.parse_args()

    if args.command == COMMANDS.EXTRACT.value:
        feature_extractor: FeatureExtractor = FeatureExtractor(
            infile=args.infile,
            biome_path=args.biome_path,
            labels_path=args.labels_path,
            temp_dir=args.temp_dir,
            save_labels=args.save_labels,
            kernel_size=args.kernel_size,
            max_distance=args.max_distance,
            ratio=args.ratio,
            convert2lab=args.convert2lab,
        )

        if args.sqlite:
            feature_extractor.save_to_sqlite(db_name=args.db_name)
        if args.csv:
            feature_extractor.save_to_csv()
    elif args.command == COMMANDS.TRAIN.value:
        ...
    elif args.command == COMMANDS.PREDICT.value:
        ...
    else:
        raise ValueError("Command not supported")


if __name__ == "__main__":
    app()
