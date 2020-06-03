import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import matplotlib.pyplot as plt


#Reading CSV file
book_df = pd.read_csv("../Data/book.csv",encoding='latin-1')
print(book_df.head())
print(book_df.shape)
book_df1= book_df.drop(book_df.columns[0], axis=1)
print (book_df1.columns)
book_df1.columns=["User_ID", "Book_Title", "Book_Rating"]

#checking for null value
print (book_df1.isnull().sum())
print(book_df1["Book_Rating"].isnull().sum())

#Visualization
counts = book_df1.Book_Rating.value_counts(sort='ascending')
print (counts)
counts.plot(kind='bar')
plt.title('Rating Distribution\n')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

# Calculate mean rating of all movies 
print (book_df1.groupby('Book_Title')['Book_Rating'].mean().sort_values(ascending=False).head())

# Calculate count rating of all movies 
print (book_df1.groupby('Book_Title')['Book_Rating'].count().sort_values(ascending=False).head())

# creating dataframe with 'rating' count values 
ratings = pd.DataFrame(book_df1.groupby('Book_Title')['Book_Rating'].mean())  
ratings['num of ratings'] = pd.DataFrame(book_df1.groupby('Book_Title')['Book_Rating'].count()) 
print (ratings)

# Sorting values according to the 'num of rating column' 
bookmat = book_df1.pivot_table(index ='User_ID', columns ='Book_Title', values ='Book_Rating') 
print(bookmat.head()) 
print(ratings.sort_values('num of ratings', ascending = False).head(10))
input()

# analysing correlation with similar movies 
The_Subtle_Knife_user_ratings = bookmat['The Subtle Knife (His Dark Materials, Book 2)'] 
The_Amber_Spyglass_user_ratings = bookmat['The Amber Spyglass (His Dark Materials, Book 3)'] 
print(The_Subtle_Knife_user_ratings.head())

# analysing correlation with similar movies 
similar_to_The_Subtle_Knife = bookmat.corrwith(The_Subtle_Knife_user_ratings)
similar_to_The_Amber_Spyglass = bookmat.corrwith(The_Amber_Spyglass_user_ratings)
  
corr_The_Subtle_Knife = pd.DataFrame(similar_to_The_Subtle_Knife, columns =['Correlation']) 
corr_The_Subtle_Knife.dropna(inplace = True) 
# print(corr_The_Subtle_Knife.head())

# Similar movies like The_Subtle_Knife
print(corr_The_Subtle_Knife.sort_values('Correlation', ascending = False).head(10))
corr_The_Subtle_Knife = corr_The_Subtle_Knife.join(ratings['num of ratings']) 
# print(corr_The_Subtle_Knife.head())
  
print(corr_The_Subtle_Knife[corr_The_Subtle_Knife['num of ratings']>=1].sort_values('Correlation', ascending = False).head())
input()

# Similar movies as of The_Amber_Spyglass 
corr_The_Amber_Spyglass_user_ratings = pd.DataFrame(similar_to_The_Amber_Spyglass, columns =['Correlation'])
corr_The_Amber_Spyglass_user_ratings.dropna(inplace = True)
  
corr_The_Amber_Spyglass_user_ratings = corr_The_Amber_Spyglass_user_ratings.join(ratings['num of ratings']) 
print (corr_The_Amber_Spyglass_user_ratings[corr_The_Amber_Spyglass_user_ratings['num of ratings']>=1].sort_values('Correlation', ascending = False).head())