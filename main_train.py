import argparse

from engine.train.train_rnn import train_rnn
from engine.train.train_sup import train_sup
from engine.train.train_cnn import train_cnn

def get_args_parser():
    parser = argparse.ArgumentParser("countingViT training")
    parser.add_argument("--train_set", default="datasets/YOCO3k/labels/train.txt")
    parser.add_argument("--config_file", default="configs/config.yaml")
    parser.add_argument("--state_dict", default="")
    parser.add_argument("--save_dir", default="weights/test_save")
    parser.add_argument("--model", type=str, default="rnn")

    return parser

def main(args):
    if args.model == 'rnn':
        train_rnn(args)
    elif args.model == 'sup':
        train_sup(args)
    elif args.model == 'cnn':
        train_cnn(args)

if __name__== "__main__":
    args = get_args_parser().parse_args()
    main(args)
