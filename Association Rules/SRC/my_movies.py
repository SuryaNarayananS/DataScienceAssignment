import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

movies_df = pd.read_csv("../Data/my_movies.csv")
print (movies_df.head())
print(movies_df.shape)

#Data PreProcessing
movie_new_df = movies_df.drop(["V1", "V2", "V3", "V4", "V5"], axis = 1)
#Removing these columns because apriori module requires a dataframe that has either 0 and 1 or True and False as data
print (movie_new_df.head())

frequent_itemsets1= apriori(movie_new_df, min_support=0.1, max_len=5,use_colnames = True)
frequent_itemsets1.sort_values('support',ascending = False,inplace=True)
print (frequent_itemsets1)


frequent_itemsets2 = apriori(movie_new_df, min_support=0.2, max_len=3,use_colnames = True)
frequent_itemsets2.sort_values('support',ascending = False,inplace=True)
print (frequent_itemsets2)

frequent_itemsets3 = apriori(movie_new_df, min_support=0.1, max_len=7,use_colnames = True)
frequent_itemsets3.sort_values('support',ascending = False,inplace=True)
print (frequent_itemsets3)


#Association rule with lift criteria
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
print (rules3)

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

