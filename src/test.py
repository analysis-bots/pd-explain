import pandas as pd
import pd_explain

spotify_all = (pd.read_csv(r"C:\Users\User\Desktop\pd_explain_test\pd-explain\Examples\Datasets\spotify_all.csv"))
grouped3 = spotify_all.groupby(['main_artist'])['energy'].mean()
grouped3.explain()
