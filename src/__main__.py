import fire
from src.cli import Cli


def main() -> None:
    """Entry point for the command-line interface."""
    fire.Fire(Cli)


if __name__ == "__main__":
    main()
