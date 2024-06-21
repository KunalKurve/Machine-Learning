import gradio as gr 
import pandas as pd
import matplotlib.pylab as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

groceries = []
with open("Groceries.csv","r") as f : groceries = f.read()
groceries = groceries.split("\n")

groceries_list = []
for i in groceries:
    groceries_list.append(i.split(","))
    
    
te= TransactionEncoder()
te_ary = te.fit(groceries_list).transform(groceries_list)

