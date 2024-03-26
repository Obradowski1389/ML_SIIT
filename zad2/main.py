import pandas as pd
import numpy as np
import scipy
import sys


def main(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
