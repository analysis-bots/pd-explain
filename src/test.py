import pandas as pd
import pd_explain
bank_all = (pd.read_csv(r"C:\Users\itaye\Desktop\pdexplain\pd-explain\Examples\Datasets\bank_churners_user_study.csv"))
by_category = bank_all.groupby(['Income_Category'])['CLIENTNUM'].count()
# print(bank_all['Education_Level'].dtype.name)
# print(by_category.explain(target='Less than $40K', explainer='outlier', dir=1))

spotify_all = (pd.read_csv(r"C:\Users\itaye\Desktop\pdexplain\pd-explain\Examples\Datasets\spotify_all.csv"))
new_songs = spotify_all[spotify_all['decade']>1970]
spotify_filtered = spotify_all[(spotify_all.loudness <= -28) & (spotify_all['decade'] <= 2000)]
spotify_filtered_grouped = spotify_filtered.groupby(['decade']).instrumentalness.mean()
print(spotify_filtered_grouped.explain(explainer='outlier',target=1950 , dir=-1))
# grouped3 = new_songs.groupby(['decade'])
# grouped3_mean_by_popularity = grouped3['popularity'].agg('mean')
# print(grouped3_mean_by_popularity.explain(explainer='outlier', target=2020, dir=-1))
pass
