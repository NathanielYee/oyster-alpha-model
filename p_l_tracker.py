import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels as sm
import scipy as sp
import seaborn as sns
import streamlit as st
import requests

import requests

def pair_data(tokenAddresses: str):
    import requests

    response = requests.get(
        url=f"https://api.dexscreener.com/latest/dex/tokens/{tokenAddresses}",
        headers={},
    )
    data = response.json()
    if response.status_code == 200:
        data = response.json()
        return data


def main():
    # Call pair_data and print the response
    data = pair_data(tokenAddresses="0x8B0E6f19Ee57089F7649A455D89D7bC6314D04e8")
    data_df = pd.DataFrame(data)
    print(data_df)


# Ensure main runs only when the script is executed directly
if __name__ == "__main__":
    main()
