from utils.utilities import split_data
from utils.utilities import cartesian_converter
import pandas as pd
import numpy as np
import pickle5 as pickle

if __name__ == "__main__":
    with open('data/pi0.pkl', 'rb') as f:
        xz = np.array(pickle.load(f), dtype=np.float64)
    dfx = pd.DataFrame(xz)

    train,test = split_data(dfx)
    print(train)
    print(test)
    train.to_pickle("data/pi0_spherical_train.pkl")
    test.to_pickle("data/pi0_spherical_test.pkl")

