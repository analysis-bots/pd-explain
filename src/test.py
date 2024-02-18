import pandas as pd
import pd_explain

credit_card_customers = (pd.read_csv(r"C:\Users\itaye\Desktop\pdexplain\pd-explain\Examples\Datasets\bank_churners_user_study.csv"))
print(type(credit_card_customers))

filter_results = credit_card_customers.where(credit_card_customers["Customer_Age"] > 45)
print("hi")

explains = filter_results.explain(top_k=1, show_scores=True)