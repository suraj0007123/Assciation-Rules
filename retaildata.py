###### Importing the library pandas and apriori , association_rules
import pandas as pd
from mlxtend.frequent_patterns import apriori , association_rules

#### Loading the movie dataset for the MBA analysis  
retaildata =[]
with open(r"E:\DESKTOPFILES\suraj\assigments\association rulesss\Datasets_Association Rules\transactions_retail1.csv") as f:
    retaildata = f.read()

### splitting the dataset into separate transactions using separator as '\n'
retaildata = retaildata.split('\n')

retaildata_list = []
for i in retaildata:
    retaildata_list.append(i.split(','))

all_retaildata_list = [ i for item in retaildata_list for i in item]

from collections import Counter
item_frequencies = Counter(all_retaildata_list)

# after sorting
item_frequencies = sorted(item_frequencies.items(), key = lambda x:x [1])

# Storing frequencies and items in separate variables 
frequencies = list(reversed([i[1] for i in item_frequencies]))
items = list(reversed([i[0] for i in item_frequencies]))

# barplot of top 10 
import matplotlib.pyplot as plt

plt.figure(figsize=(18,8))
plt.bar(height = frequencies[0:11], x=list(range(0,11)), color=['r','g','b','k','y','m','c'])
plt.xticks(list(range(0, 11),), items[0:11])
plt.title('The Frequencies of retaildata')
plt.xlabel('items')
plt.ylabel('Count')
plt.show()

# Creating Data Frame for the transactions data
retaildata_series = pd.DataFrame(pd.Series(retaildata_list))
retaildata_series = retaildata_series.iloc[:557040, :] # removing the last empty transaction

retaildata_series.columns = ['trans']

# creating a dummy columns for the each item in each transactions ... Using column names as item name
X = retaildata_series['trans'].str.join(sep = '*').str.get_dummies(sep = '*')

# Most Frequent item sets based on support
frequent_itemsets = apriori(X, min_support = 0.0075, max_len=4, use_colnames=True)

plt.figure(figsize=(18,8))
plt.bar(x = list(range(0,11)),height = frequent_itemsets.support[0:11],color=['r','g','m','y','k'])
plt.xticks(list(range(0,11)),frequent_itemsets.itemsets[0:11], rotation=20)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.sort_values('lift',ascending = False,inplace=True)

######################Eliminating the reducdencies in the rules#######################
def to_list(i):
    return (sorted(list(i)))

#Sorting, listing and appending

ma_X = rules.antecedents.apply(to_list)+rules.consequents.apply(to_list)

ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]

index_rules = []
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))

# getting rules without any redudancy 
rules_no_redudancy  = rules.iloc[index_rules,:]

# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('lift',ascending=False).head(10)

plt.figure(figsize=(18,8))
plt.bar(x = list(range(0,11)),height = rules_no_redudancy.lift[0:11],color=['r','g','m','y','k'])
plt.xticks(list(range(0,11)),rules_no_redudancy.antecedents[0:11], rotation=20)



######## Insight #####
#### The coats are most purchased item with other items like holder & hanging ,Wooly , polka & red , Cream 
##### Dot , Nesting & Boxes 
##### Has i have mentioned above products those should have beside each-other, or near to one another

