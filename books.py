###### Importing the library pandas and apriori , association_rules
import pandas as pd
from mlxtend.frequent_patterns import apriori , association_rules

#### Loading the book dataset for the MBA analysis  
book=[]
with open(r"E:\DESKTOPFILES\suraj\assigments\association rulesss\Datasets_Association Rules\book.csv") as f:
    book=f.read()

### splitting the dataset into separate transactions using separator as '\n'
book = book.split('\n')


book_list = []
for i in book:
     book_list.append(i.split(','))
     
all_book_list = [i for item in book_list for i in item]

from collections import Counter

item_frequencies = Counter(all_book_list)

# after sorting
item_frequencies = sorted(item_frequencies.items(), key = lambda x:x[1])

# Storing frequencies and items in separate variables 
frequencies = list(reversed([i[1] for i in item_frequencies]))

items = list(reversed([i[0] for i in item_frequencies]))

# barplot of top 10 
import matplotlib.pyplot as plt

plt.figure(figsize=(18,8))
plt.bar(height = frequencies[0:11], x = list(range(0,11)), color = ['r','g','b','k','y','m','c'])
plt.xticks(list(range(0,11), ), items[0:11])
plt.xlabel("items")
plt.ylabel("Count")
plt.show()

# Creating Data Frame for the transactions data
book_series = pd.DataFrame(pd.Series(book_list))
book_series = book_series.iloc[:2000, :] # removing the last empty transaction

book_series.columns = ['trans']

# creating a dummy columns for the each item in each transactions ... Using column names as item name
X = book_series['trans'].str.join(sep = '*').str.get_dummies(sep = '*')

# Most Frequent item sets based on support 
frequent_itemsets = apriori(X, min_support = 0.0075, max_len=4, use_colnames=True)

plt.figure(figsize=(18,8))
plt.bar(x = list(range(0,11)), height = frequent_itemsets.support[0:11], color =['r','g','m','y','k'])
plt.xticks(list(range(0,11)), frequent_itemsets.itemsets[0:11], rotation=20)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()

rules = association_rules(frequent_itemsets, metric="lift",min_threshold=1)
rules.head(10)
rules.sort_values('lift', ascending=False).head(10)

######################Eliminating the reducdencies in the rules#######################
def to_list(i):
    return(sorted(list(i)))

ma_x = rules.antecedents.apply(to_list)+rules.consequents.apply(to_list)

ma_x = ma_x.apply(sorted)

return_rules = list(ma_x)

unique_rules = [list(m) for m in set(tuple(i) for i in return_rules)]        

index_rules = []
for i in unique_rules:
    index_rules.append(return_rules.index(i))
        
    
##Getting the rules without any redudancy
rules_no_redudancy = rules.iloc[index_rules, : ]

#Sorting them with respect to lift 
rules_no_redudancy.sort_values('lift', ascending = False, inplace = True)

rules.head(10)

plt.figure(figsize=(18,12))
plt.bar(x = list(range(0,11)),height = rules_no_redudancy.lift[0:11],color=['r','g','m','y','k'])
plt.xticks(list(range(0,11)),rules_no_redudancy.antecedents[0:11],rotation=30)



##### Insights 
### has we see in the dataset that cookbooks and childsbooks , doltybooks are the most frequently purchased  
### books by the customer so that the owner of kitabi duniya book store want to be event the books exhibition so that public we aware of books will to same new books 

 
