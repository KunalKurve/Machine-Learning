import gradio as gr 
import pandas as pd
import matplotlib.pylab as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from warnings import filterwarnings

filterwarnings('ignore')

fp_df = pd.read_csv('Cosmetics.csv',index_col=0)
fp_df = fp_df.astype(bool)

def gen_rules(min_sup, min_conf):
    itemsets = apriori(fp_df, min_support=min_sup, use_colnames=True)
    rules = association_rules(itemsets, metric='confidence', min_threshold=min_conf)
    rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    rules['antecedents'] = [list(x) for x in rules['antecedents'].values]
    rules['consequents'] = [list(x) for x in rules['consequents'].values]
    rules = rules[rules['lift']>1]
    return rules.sort_values(by='lift', ascending=False)

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