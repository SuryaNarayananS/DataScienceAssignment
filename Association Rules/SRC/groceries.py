import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

groceries = []
with open("../Data/groceries.csv") as f:
    groceries = f.read()
    groceries = groceries.split("\n")

groceries_list = []
for i in groceries:
	groceries_list.append(i.split(","))
	

print (groceries_list)
input()

all_groceries_list = [i for item in groceries_list for i in item]
print (all_groceries_list)
input()

#Data PreProcessing
from collections import Counter
item_frequencies = Counter(all_groceries_list)
# print (item_frequencies)
item_frequencies = sorted(item_frequencies.items(),key = lambda x:x[1])
# print (item_frequencies)

# # Storing frequencies and items in separate variables 
frequencies = list(reversed([i[1] for i in item_frequencies]))
items = list(reversed([i[0] for i in item_frequencies]))

# print (frequencies)
# print (items)
import matplotlib.pyplot as plt

plt.bar(height = frequencies[0:11],x = list(range(0,11)),color='rgbkymc');
plt.xticks(list(range(0,11),),items[0:11]);
plt.xlabel("items")
plt.ylabel("Count")
plt.show()

#creating dataframe on the transaction list
groceries_series  = pd.DataFrame(pd.Series(groceries_list))
groceries_series = groceries_series.iloc[:9835,:] # removing the last empty transaction from the data set

#Data PreProcessing
# creating a dummy column for the each item in each transactions ... Using column names as item name
groceries_series.columns = ["transactions"]
X = groceries_series['transactions'].str.join(sep='$').str.get_dummies(sep='$')
print(X)

#Apriori Algorithm
frequent_itemsets1 = apriori(X, min_support=0.005, max_len=3,use_colnames = True)
frequent_itemsets1.sort_values('support',ascending = False,inplace=True)
print (frequent_itemsets1)

frequent_itemsets2 = apriori(X, min_support=0.005, max_len=7,use_colnames = True)
frequent_itemsets2.sort_values('support',ascending = False,inplace=True)
print (frequent_itemsets2)

frequent_itemsets3 = apriori(X, min_support=0.1, max_len=3,use_colnames = True)  #There is only one itemsets for support = 0.1
frequent_itemsets3.sort_values('support',ascending = False,inplace=True)
print (frequent_itemsets3)

frequent_itemsets4 = apriori(X, min_support=0.15, max_len=5,use_colnames = True) #There is only one itemsets for support = 0.15
frequent_itemsets4.sort_values('support',ascending = False,inplace=True)
print (frequent_itemsets4)

# Visualization of Most Frequent item sets based on support 
plt.bar(x = list(range(1,11)),height = frequent_itemsets1.support[1:11],color='rgmyk');
plt.xticks(list(range(1,11)),frequent_itemsets1.itemsets[1:11])
plt.xlabel('item-sets');
plt.ylabel('support')
plt.show()

#Rules
rules1 = association_rules(frequent_itemsets1, metric="lift", min_threshold=1)
rules1.sort_values('lift',ascending = False,inplace=True)
print (rules1)

#Visualization for rules1
plt.scatter(rules1['support'], rules1['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()

plt.scatter(rules1["support"], rules1["lift"])
plt.xlabel("support")
plt.ylabel("lift")
plt.title("Support vs Lift")
plt.show()



rules2 = association_rules(frequent_itemsets2, metric="lift", min_threshold=1)
rules2.sort_values('lift',ascending = False,inplace=True)
print (rules2)

#Visualization for rules2
plt.scatter(rules2['support'], rules2['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()

plt.scatter(rules2["support"], rules2["lift"], alpha=0.5)
plt.xlabel("support")
plt.ylabel("lift")
plt.title("Support vs Lift")
plt.show()



rules3 = association_rules(frequent_itemsets3, metric="lift", min_threshold=1)
rules3.sort_values('lift',ascending = False,inplace=True)
print (rules3)   #No Antecedent and consequents as there is only on itemset for support = 0.1

#Visualization for rules3
plt.scatter(rules3['support'], rules3['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()

plt.scatter(rules3["support"], rules3["lift"], alpha=0.5)
plt.xlabel("support")
plt.ylabel("lift")
plt.title("Support vs Lift")
plt.show()



rules4 = association_rules(frequent_itemsets4, metric="lift", min_threshold=1)
rules4.sort_values('lift',ascending = False,inplace=True)
print (rules4)     #No Antecedent and consequents as there is only on itemset for support = 0.1

#Visualization for rules4
plt.scatter(rules4['support'], rules4['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()

plt.scatter(rules4["support"], rules4["lift"], alpha=0.5)
plt.xlabel("support")
plt.ylabel("lift")
plt.title("Support vs Lift")
plt.show()