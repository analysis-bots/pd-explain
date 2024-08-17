import pandas as pd
import pd_explain
# bank_all = (pd.read_csv(r"C:\Users\itaye\Desktop\pdexplain\pd-explain\Examples\Datasets\bank_churners_user_study.csv"))
# low_income = bank_all[bank_all['Income_Category'] == 'Less than $40K']
# by_gender = low_income.groupby(['Gender']).CLIENTNUM.count()
# by_gender.explain(explainer='outlier', target='F', dir=1)

# by_category = bank_all.groupby(['Income_Category', 'Gender'])['CLIENTNUM'].count()
# print(bank_all['Education_Level'].dtype.name)
# print(by_category.explain(target=('Less than $40K', 'F'), explainer='outlier', dir=1))

songs_df = (pd.read_csv(r"C:\Users\itaye\Desktop\pdexplain\pd-explain\Examples\Datasets\spotify_all.csv"))
# t = type(songs_df[['decade']])
# popular_songs_df = songs_df[songs_df['popularity'] > 65]
# popular_songs_df.explain()

new_songs = songs_df[songs_df['decade']>1970]
# spotify_filtered = spotify_all[(spotify_all.loudness <= -28) & (spotify_all['decade'] <= 2010)]
# spotify_filtered_grouped = spotify_filtered.groupby(['decade']).popularity.mean()
# spotify_filtered_grouped.explain()
# print(spotify_filtered_grouped.explain(explainer='outlier',target=1940 , dir=-1, control=[1930]))
grouped3 = new_songs.groupby('decade')
grouped3_mean_by_popularity = grouped3['popularity'].agg('mean')
(grouped3_mean_by_popularity.explain(explainer='outlier', target=2020, dir='low'))
# explicit = spotify_all[spotify_all['explicit'] != 0]
# print(explicit['decade'].value_counts())


# explicit = spotify_all[(spotify_all['explicit'] == 0) & (spotify_all['decade'] >= 1950) & (spotify_all['decade'] <= 2010)]
# explicit_by_decade = explicit.groupby(['decade']).id.count()
# print(explicit_by_decade.explain(explainer='outlier', target=2000, dir=-1))

# adults = (pd.read_csv(r"C:\Users\itaye\Desktop\pdexplain\pd-explain\Examples\Datasets\adult.csv"))
# _, bins = pd.cut(adults['capital-gain'], 10, retbins=True)
# bins = [round(b) for b in bins]
# adults['capital-gain-bins'] = pd.cut(adults['capital-gain'], bins=bins)
# adults['capital-gain-bins'] = adults['capital-gain-bins'].apply(lambda x: str(x))
# by_capital_gain = adults.groupby(['capital-gain-bins']).label.count()
# print(adults.columns)
# print(by_capital_gain.explain(explainer='outlier', target='(89999, 99999]', dir=1, hold_out=['capital-loss']))





#####################################################################################
# count_artist = songs_df.groupby('main_artist').main_artist.count()
# count_artist = count_artist[count_artist.values > 100]
# spotify_frequent = songs_df[songs_df['main_artist'].isin(count_artist.index)]

# pop_by_artist = spotify_frequent.groupby('main_artist')['popularity'].mean()
# pop_by_artist_df = pd.DataFrame({'main_artist': pop_by_artist.index, 'mean_popularity': pop_by_artist.values})
# pop_by_artist = pop_by_artist[pop_by_artist.values > 60] 

# acoustic = spotify_frequent[spotify_frequent['acousticness'] > 0.95]
# joined = acoustic.join(pop_by_artist, on='main_artist')
# joined.explain(explainer='shapley')
pass
