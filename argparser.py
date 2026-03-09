import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--epochs",type=int,default=100)
parser.add_argument("--lr",type=float,default=0.01)

parser.add_argument("--window-size",type=int,default=4)
parser.add_argument("--embedding-size",type=int,default=100)
parser.add_argument("--num-tokens",type=int,default=10000)

parser.add_argument("--negative-sampling",action="store_true")
parser.add_argument("--k",type=int,default=4)