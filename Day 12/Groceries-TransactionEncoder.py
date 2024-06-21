import gradio as gr 
import pandas as pd
import matplotlib.pylab as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from warnings import filterwarnings

filterwarnings('ignore')

groceries = []
with open("Groceries.csv","r") as f : groceries = f.read()
groceries = groceries.split("\n")

groceries_list = []
for i in groceries:
    groceries_list.append(i.split(","))
    
    
te= TransactionEncoder()
te_ary = te.fit(groceries_list).transform(groceries_list)

fp_df = pd.DataFrame(te_ary, columns = te.columns_)

## Support of 1-tem freq sets

itemFrequency = fp_df.sum(axis = 0 )/ len(fp_df)
itemFrequency = itemFrequency.sort_values(ascending=False)

# and plot as histogram

ax = itemFrequency[:10].plot.barh(color = "blue")
plt.ylabel('Item Frequency (relative) of Top 10')
plt.show()

itemFrequency = itemFrequency.sort_values(ascending=True)

# and plot as histogram

ax = itemFrequency[:10].plot.barh(color = "blue")
plt.ylabel('Item Frequency (relative) of Top 10')
plt.show()

#create frequent itemsets
itemsets = apriori(fp_df, min_support=0.01, use_colnames=True)

def gen_rules(min_sup, min_conf):
    itemsets = apriori(fp_df, min_support=min_sup, use_colnames=True)
    rules = association_rules(itemsets, metric='confidence', min_threshold=min_conf)
    rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    rules['antecedents'] = [list(x) for x in rules['antecedents'].values]
    rules['consequents'] = [list(x) for x in rules['consequents'].values]
    rules = rules[rules['lift']>1]
    rules.sort_values(by='lift', ascending=False)
    return rules

demo = gr.Interface(gen_rules, 
                    inputs= [gr.Slider(value=0.01, step=0.01,
                                   label="Minimum Support",
                                   minimum=0.0001, maximum=1),
                             gr.Slider(value=0.01, step=0.01,
                                   label="Minimum Confidence",    
                                   minimum=0.0001, maximum=1)], 
                    outputs='dataframe')

if __name__ == "__main__":
    demo.launch()