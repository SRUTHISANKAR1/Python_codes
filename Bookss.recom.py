#######Recommendation system -Book.csv###########
import pandas as pd

Book=pd.read_csv("C:/Users/server/Downloads/book (2).csv",encoding="ISO-8859-1")
for col in Book.columns: 
    print(col) 
Books = Book.drop(columns="Unnamed: 0")

Books.shape  #1000obs and 3 variables
Books.columns

Books =Books.rename(columns={'Book.Title': 'Title','User.ID':'ID','Book.Rating':'Rating'}) #for convenience I renamed the columns

#create a user matrix or rating matrix based on books
#import tfidf matrix
from sklearn.feature_extraction.text import TfidfVectorizer#Term frequency inverse document frequency (Tfidf)is a numerical statistics that is intended to reflect how important a word is to document in a collection and corpus
#creating a Tfidf Vectorizer to remove all stopwords
tfidf=TfidfVectorizer(stop_words="english")


#check for null values(NaN)
Books["Title"].isnull().sum()    #zero null values


#transform Books["Book.Title"]into a matrix format
tfidf_matrix=tfidf.fit_transform(Books.Title)
tfidf_matrix.shape    #(10000, 11435)

#there are several similarity matrix or scores 
#here we use cosine similarity scores(matrix). 
#cosine similarity metric is independent of magnitude and easy to calculate
#cosine(x,y)=(x,yT)/(||X||.||Y||)

#find cosine similarity between tfidf vs tfidf matrix
from sklearn.metrics.pairwise import linear_kernel
cosine_sim_metrics=linear_kernel(tfidf_matrix,tfidf_matrix)

#creating a mapping of Title to index number
#drop duplicates
Books_index=pd.Series(Books.index,index=Books["Title"]).drop_duplicates()

#just check the index
Books_index["PLEADING GUILTY"]    #7

#execute line number 47 to 67  together
#identify similarity for particular Anime
def get_Books_recommendations(Title,topN):   #topN is top N number of recommendations- if topN=top10
    Books_id=Books_index[Title]  #getting the book index using its title
#for example PLEADING GULITY is stored and get id for that
#and get the index number 7(PLEADING GULITY ) and 7  pair wise similarity score for all Book vs Book
    cosine_scores=list(enumerate(cosine_sim_metrics[Books_id]))
#sorting cosine similarity scores based on scores by using lambda function - in descending order
    cosine_scores=sorted(cosine_scores,key=lambda x:x[1],reverse=True)
#out of the total observations  get the scores of topN+1 observations(or top 10 most similar books)
#topN+1 means if we want top 10 observation(index 0 to 11 index then only we get output 10)
    cosine_scores_10=cosine_scores[0:topN+1]
#get the Book index
    Books_idx=[i[0] for i in cosine_scores_10]
    Books_scores=[i[1] for i in cosine_scores_10]

#creating data frame of similar Books and scores with two columns name and score
    Books_similar=pd.DataFrame(columns=["Title","score"])
    Books_similar["Title"]=Books.loc[Books_idx,"Title"]
    Books_similar["score"]=Books_scores
    Books_similar.reset_index(inplace=True) #resetting index
    Books_similar.drop(["index"],axis=1,inplace=True) #resetting index
    print(Books_similar)
#return Books_similar_show
#execute altogether    


#enter your Anime & number of Books to be recommended
get_Books_recommendations("PLEADING GUILTY",topN=5)

get_Books_recommendations("Jane Doe",topN=5)

get_Books_recommendations("Scarlet Letter",topN=5) 

get_Books_recommendations("The Middle Stories",topN=5)
