from config import hparams
from train import Trainer
from test import Tester
from vocab import dump_vocab
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    default_attrs = vars(hparams)
    for k, v in default_attrs.items():
        parser.add_argument('--' + str(k), default=v)
    args = parser.parse_args()
    # print(vars(args))
    dump_vocab(args.vocab_root, args.train_src_path, args.train_dst_path, int(args.threshold))
    trainer = Trainer(args)
    trainer.train()
    tester = Tester(args)
    tester.test()
