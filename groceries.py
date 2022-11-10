## Importing the pandas library and implementing the apriori algorithm from mlxtend
import pandas as pd
from mlxtend.frequent_patterns import apriori , association_rules

#### Loading the groceries dataset for MBA analysis
groceries = []
with open(r"E:\DESKTOPFILES\suraj\assigments\association rulesss\Datasets_Association Rules\groceries.csv") as f:
    groceries = f.read()
    
#### Splitting the data into separate transactions using separator '\n'
groceries = groceries.split('\n')


groceries_list = []
for i in groceries:
    groceries_list.append(i.split(','))
    
all_groceries_list = [i for item in groceries_list for i in item]

from collections import Counter

item_frequencies =  Counter(all_groceries_list)

# after sorting
item_frequencies = sorted(item_frequencies.items(), key = lambda x:x[1])

# Storing frequencies and items in separate variables 
frequencies = list(reversed([i[1] for i in item_frequencies]))
items = list(reversed([i[0] for i in item_frequencies]))

# barplot of top 10
import matplotlib.pyplot as plt

plt.figure(figsize=(18,8));plt.bar(height = frequencies[0:11], x = list(range(0,11)), color=['r','g','b','k','y','m','c'])
plt.xticks(list(range(0,11),), items[0:11])
plt.xlabel("items")
plt.ylabel("Count")
plt.show()

# Creating Data Frame for the transactions data
groceries_series = pd.DataFrame(pd.Series(groceries_list))
groceries_series = groceries_series.iloc[0:9835, :]

groceries_series.columns = ['trans']

# creating a dummy columns for the each item in each transactions ... Using column names as item name
X =  groceries_series['trans'].str.join(sep = '*').str.get_dummies(sep = '*')

frequent_itemsets = apriori(X, min_support = 0.0075, max_len=4, use_colnames=True)

# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending=False, inplace=True)

import matplotlib.pyplot as plt

plt.figure(figsize=(18,8));plt.bar(x=list(range(0,11)), height=frequent_itemsets.support[0:11], color=['r','g','b','k','y','m','c'])
plt.xticks(list(range(0,11),), frequent_itemsets.itemsets[0:11], rotation=20)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()

rules = association_rules(frequent_itemsets,  metric='lift', min_threshold=1)
rules.head(10)
rules.sort_values('lift', ascending=False).head(10)

#To eliminate Redudancy in Rules
def to_list(i):
    return (sorted(list(i)))

#Sorting, listing and appending
ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)

ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]

index_rules = []
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))

# getting rules without any redudancy 
rules_no_redudancy = rules.iloc[index_rules, :]

# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('lift', ascending = False).head(10)

plt.figure(figsize=(18,12))
plt.bar(x = list(range(0,11)),height = rules_no_redudancy.lift[0:11],color=['r','g','m','y','k'])
plt.xticks(list(range(0,11)),rules_no_redudancy.antecedents[0:11],rotation=30)



######### Insight :- 
## Has we see the above barplot that barplot tells us the whole milk associated with lots of other products like root vegetables 
## and rolls and buns so therefore the shop should launch same offers like giving the discounts that it may increase sells and develop the business 