import requests   # Importing requests to extract content from a url
from bs4 import BeautifulSoup as bs # Beautifulsoup is for web scrapping...used to scrap specific content 
import re 
import pandas as pd
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from wordcloud import WordCloud

### Extracting reviews from Amazon website ################

url="https://www.amazon.in/Test-Exclusive-749/product-reviews/B07DJ8K2KT/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews"
response = requests.get(url)
soup = bs(response.content,"html.parser")# creating soup object to iterate over the extracted content
# print (soup.prettify())

#scrapping user name
name= soup.find_all("span",class_="a-profile-name")
print(name) 
user_name=[]
for i in range(0,len(name)):
  user_name.append(name[i].get_text())
# print(user_name)
user_name.pop(0)      #popping first two as it is repeating
user_name.pop(0)
print (user_name)
print (len(user_name))


#scrapping review title
review= soup.find_all("a",class_="review-title-content")
# print(review)
review_title=[]
for i in range(0,len(review)):
  review_title.append(review[i].get_text())
# print(review_title)
review_title[:] = [titles.lstrip("\n") for titles in review_title]
review_title[:] = [titles.rstrip("\n") for titles in review_title]
print (review_title)
print (len(review_title))


#scrapping review ratings
rating= soup.find_all("i",{"data-hook":"review-star-rating"})
# print(review)
review_ratings=[]
for i in range(0,len(rating)):
  review_ratings.append(rating[i].get_text())
print(review_ratings)
print(len(review_ratings))

#scrapping review content
content = soup.find_all("span",{"data-hook":"review-body"})
review_content=[]
for i in range(0,len(content)):
  review_content.append(content[i].get_text())
# print(review_content)
review_content[:] = [content.lstrip("\n") for content in review_content]
review_content[:] = [content.rstrip("\n") for content in review_content]
print (review_content)
print (len(review_content))

#creating dataframe and saving it in CSV file
df=pd.DataFrame()
df["user_name"] = user_name
df["review_title"] = review_title
df["review_ratings"] = review_ratings
df["review_content"] = review_content

print (df.head())
# df.to_csv(r'D:/Data Science Course/Assignments/Text Mining/Amazon_Reviews.csv', index=True)

# Joinining all the reviews into single paragraph
review_content_string = " ".join(review_content)
# print(review_content_string)
# input()


# Removing unwanted symbols incase if exists
review_content_string = re.sub("[^A-Za-z" "]+"," ",review_content_string).lower()
review_content_string = re.sub("[0-9" "]+"," ",review_content_string)
# print (review_content_string)
# input()


# words that contained in OnePLus reviews
review_content_words = review_content_string.split(" ")

stop_words = stopwords.words('english')

with open("../Data/stop.txt","r") as sw:
    stopwords = sw.read()

stopwords = stopwords.split("\n")

review_content_words = [w for w in review_content_words if not w in stopwords]



# Joinining all the reviews into single paragraph 
review_content_string = " ".join(review_content_words)

# WordCloud can be performed on the string inputs. That is the reason we have combined 
# entire reviews into single paragraph
# Simple word cloud


wordcloud_review_content = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(review_content_string)

plt.imshow(wordcloud_review_content)
plt.show()

# positive words # Choose the path for +ve words stored in system
with open("../Data/positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")
  
poswords = poswords[36:]



# negative words  Choose path for -ve words stored in system
with open("../Data/negative-words.txt","r") as neg:
  negwords = neg.read().split("\n")

negwords = negwords[37:]

# negative word cloud
# Choosing the only words which are present in negwords
review_content_neg_in_neg = " ".join ([w for w in review_content_words if w in negwords])

wordcloud_neg_in_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(review_content_neg_in_neg)

plt.imshow(wordcloud_neg_in_neg)
plt.show()

# Positive word cloud
# Choosing the only words which are present in positive words
review_content_pos_in_pos = " ".join ([w for w in review_content_words if w in poswords])
wordcloud_pos_in_pos = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(review_content_pos_in_pos)

plt.imshow(wordcloud_pos_in_pos)
plt.show()


# Unique words 
review_content_unique_words = list(set(" ".join(review_content).split(" ")))
print(review_content_unique_words)

