import argparse
import sys
from src.phishguard.ml.train import train
from src.phishguard.ml.evaluate import evaluate
from src.phishguard.data.prepare_dataset import prepare_dataset as prepare_dataset



def main():
    parser = argparse.ArgumentParser(
        prog="phishguard",
        description="PhishGuard is a tool to detect phishing domains in your website."
        )

    #The one excat positional argument expected:
    parser.add_argument("command",
                        #Allowing no arguments launch
                        nargs="?",
                        choices =["train", "evaluate", "prepare-dataset"],
                        help="which action to perform"
                        )

    #Extra argument:
    parser.add_argument(
        "command_args",
        nargs=argparse.REMAINDER,
        help = "command arguments"
    )

    args = parser.parse_args()

    if args.command == None:
        print("Please specify a command:")
        print("train")
        print("evaluate")
        sys.exit(0)

    cmd_args = args.command_args
    if cmd_args and cmd_args[0] == "--":
        cmd_args = cmd_args[1:]
    if args.command == "prepare-dataset":
        prepare_dataset(cmd_args)
        sys.exit(0)
    elif args.command == "train":
        train()
        sys.exit(0)
    elif args.command == "evaluate":
        evaluate()
        sys.exit(0)


if __name__ == "__main__":
    main()