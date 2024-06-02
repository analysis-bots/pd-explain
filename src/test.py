import pandas as pd
import pd_explain
spotify_all = (pd.read_csv(r"C:\Users\itaye\Desktop\pdexplain\pd-explain\Examples\Datasets\spotify_all.csv"))
new_songs = spotify_all[spotify_all['decade']>1970]
grouped3 = new_songs.groupby(['decade'])
grouped3_by_popularity=grouped3['popularity']
grouped3_mean_by_popularity = grouped3_by_popularity.mean()
# grouped3_mean_by_popularity = new_songs.groupby(['decade'])
# print(grouped3_mean_by_popularity[2020])
print(grouped3_mean_by_popularity.explain(explainer='outlier', target=2020))
pass
