import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

house = pd.read_csv('Housing.csv')
intervals = [(0, 50000), (50000, 100000), (100000, 150000), (150000, 200000)]

bins1 = pd.IntervalIndex.from_tuples(intervals)
house['price_slab'] = pd.cut(house['price'], bins1)

intervals = [(0, 5000), (5000, 10000), (10000, 15000), (15000, 20000)]

bins2 = pd.IntervalIndex.from_tuples(intervals)
house['area_slab'] = pd.cut(house['lotsize'], bins2)

print(house['price_slab'].value_counts())
print(house['area_slab'].value_counts())

house.drop(['price', 'lotsize'], axis=1, inplace = True)

house = house.astype(object)
fp_df = pd.get_dummies(house, prefix_sep='=')

itemFrequency = fp_df.sum(axis = 0 )/ len(fp_df)
itemFrequency = itemFrequency.sort_values(ascending=False)

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