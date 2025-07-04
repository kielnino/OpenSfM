# pyre-strict
import argparse

from opensfm.actions import export_bundler
from opensfm.dataset import DataSet

from . import command


class Command(command.CommandBase):
    name = "export_bundler"
    help = "Export reconstruction to bundler format"

    def run_impl(self, dataset: DataSet, args: argparse.Namespace) -> None:
        export_bundler.run_dataset(
            dataset, args.list_path, args.bundle_path, args.undistorted
        )

    def add_arguments_impl(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--list_path", help="path to the list.txt file")
        parser.add_argument("--bundle_path", help="path to the bundle.out file")
        parser.add_argument(
            "--undistorted",
            action="store_true",
            help="export the undistorted reconstruction",
        )
