import pandas as pd
import pd_explain

credit_card_customers = (pd.read_csv(r"C:\Users\itaye\Desktop\pdexplain\pd-explain\Examples\Datasets\bank_churners_user_study.csv"))
# print(type(credit_card_customers))
spotify_all = (pd.read_csv(r"C:\Users\itaye\Desktop\pdexplain\pd-explain\Examples\Datasets\spotify_all.csv"))
spotify_all['main_artist'] = spotify_all['main_artist'].replace({'\$': ''}, regex=True)
grouped3 = spotify_all.groupby(['main_artist'])['popularity'].mean()

grouped3.explain()
# filter_results = credit_card_customers.where(credit_card_customers["Customer_Age"] > 45)
# explains = filter_results.explain(attributes=["Months_on_book", "Dependent_count"], show_scores=True)
# explains = filter_results.explain(top_k=2, show_scores=True)
# grouped_bank = credit_card_customers.groupby(['Marital_Status'])['Customer_Age'].mean()
# grouped_bank.explain()
print("hi")

# explains = filter_results.explain(top_k=1, show_scores=True)