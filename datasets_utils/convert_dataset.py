"""Generic CLI entrypoint that dispatches dataset conversion to the selected adapter."""

import argparse

from datasets_utils.dataset_factory import get_dataset_adapter, get_registered_dataset_types


def parse_args():
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument(
        "dataset_type",
        choices=get_registered_dataset_types(),
        help="Dataset family to convert.",
    )
    known_args, _ = base_parser.parse_known_args()

    adapter = get_dataset_adapter(known_args.dataset_type)
    parser = argparse.ArgumentParser(
        description="Convert a dataset into the flat HDF5 format expected by this trainer."
    )
    parser.add_argument(
        "dataset_type",
        choices=get_registered_dataset_types(),
        help="Dataset family to convert.",
    )
    adapter.add_conversion_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    adapter = get_dataset_adapter(args.dataset_type)
    adapter.convert_from_args(args)


if __name__ == "__main__":
    main()
