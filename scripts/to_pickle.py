import sys
import argparse

import pandas as pd

def main(argv):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("pkl_file_name",
                        help = ".pkl file you wish to convert to a csv",
                        default = "Ligands.pkl")
    parser.add_argument("csv_file_name",
                        help = "csv file name to convert to, be careful not to overwrite any other files",
                        default = "output.csv")

    if len(argv) == 0:
        print(parser.print_help())
        return

    args = parser.parse_args(argv)
    
    df = pd.read_csv(f"{args.csv_file_name}", index_col = False)

    df.to_pickle(args.pkl_file_name, protocol = 4)

if __name__  == "__main__":
    main(sys.argv[1:])