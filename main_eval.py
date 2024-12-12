import argparse

from engine.eval.eval_rnn import eval_rnn
from engine.eval.eval_rnn_sup import eval_rnn_sup
from engine.eval.eval_cnn import eval_cnn

def get_args_parser():
    parser = argparse.ArgumentParser("countingViT evaluation")
    parser.add_argument("--val_set", default="datasets/YOCO3k/labels/val.json")
    parser.add_argument("--config_file", default="configs/config.yaml")
    parser.add_argument("--state_dict", default="weights/rnn/sorted.pth")
    parser.add_argument("--visualize", default=False)
    parser.add_argument("--vis_dir", default=None)
    parser.add_argument("--model", default="rnn")

    return parser

def main(args):
    if args.model == 'rnn':
        eval_rnn(args)
    elif args.model == 'sup':
        eval_rnn_sup(args)
    elif args.model == 'cnn':
        eval_cnn(args)

if __name__== "__main__":
    args = get_args_parser().parse_args()
    main(args)
