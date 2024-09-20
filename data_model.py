#data_model.py

import numpy as np
import pandas as pd
import nltk
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
#might need this
#nltk.download('stopwords')

stop_words = set(stopwords.words('english')) # import list of english stopwords 
 

#Pre-processing
#--------------

#I commented out a lot of the preprocessing to shorten the run time
#this is possible as i saved the pre-processed csv as a new csv called simplifed.csv
#so the simplified csv only contains the pre-processed information - so do not need to pre-process the data every time


#read csv file - skip first row which is title of each column
#text_data = pd.read_csv('/Users/ameera/Desktop/m1m1/text2music/text2music/data/text.csv', header=None, skiprows=1, sep=",", encoding="utf-8", dtype={2: int})

#remove null values
#text_data.dropna(inplace=True)

#remove duplicates
#text_data.drop_duplicates(subset=[1], inplace=True)

#remove special charcaters from text
#text_data[1] = text_data[1].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))

#Lowercase the text
#text_data[1] = text_data[1].str.lower()

#remove stopwords and punctuation by tokenising the sentence into single words and spaces and rejoing the words left
#text_data[1] = text_data[1].apply(nltk.word_tokenize)
#text_data[1] = text_data[1].apply(lambda tokens: [token for token in tokens if token not in stop_words])# stopwords
#text_data[1] = text_data[1].apply(lambda tokens: [token for token in tokens if token not in string.punctuation])#punctuation
#text_data[1] = text_data[1].apply(' '.join) #rejoin sentence

#save the updated csv as a new csv to remember the correct one
#text_data.to_csv('simplified.csv', index=False, header=False)


#Training
#--------

#read the simplified CSV file
simple_df = pd.read_csv('simplified.csv', header=None)

#extract features (X) and target variable (y)
X = simple_df[1]  #features (sentences)
y = simple_df[2]  #target variable (emotion)

#fill any missing values in X - should have been removed before but just in case
X = X.fillna('')

#split the dataset into training (80% of values) and testing (20% of values) sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#initialize logistic regression classifier - max iteration 200 as the dataset is very large
logistic = LogisticRegression(max_iter=200, random_state=42)

#applying Bag of Words processing by using a count vectoriser 
count_vectorizer = CountVectorizer()

#transform text data into count vectors - split each sentence into list of unique words and keep a tally of how many times each word appears in sentence
#sparse matrix
training_data = count_vectorizer.fit_transform(X_train)
testing_data = count_vectorizer.transform(X_test)

#fit the logistic regression model to the training data
#training - learns patterns in the training set - what words appear most frequently in a sentence classified as specific emotion
logistic.fit(training_data, y_train)


#Evaluation stage - make predictions and calcualte accuracy by compairing correctly predicted to total predictions
predictions = logistic.predict(testing_data)
print('Accuracy score:', accuracy_score(y_test, predictions))

#predict emotion label from text input
def predict_emotion(text):
    
    # Preprocess the input text
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower()) #special characters
    tokens = nltk.word_tokenize(text) #split text into seperate words
    tokens = [token for token in tokens if token not in stop_words] #remove stopwords
    tokens = [token for token in tokens if token not in string.punctuation] #remove punctuation 
    text = ' '.join(tokens) #rejoin remaining words in a sentence
    
    #vectorize the preprocessed text into sparse matrix
    text_vectorized = count_vectorizer.transform([text])
    
    #predict the emotion label using logisitic regression classifier
    predicted_label = logistic.predict(text_vectorized)
    
    #return the corresponding emotion word and a short prompt based on that emotion
    #the music generator works bettwer with more detialed prompts - so return a prompt based on that emotion
    if predicted_label[0] == 0:
        print('sadness')
        return 'A melancholic melody perfect for sad, reflective moments.'
    elif predicted_label[0] == 1:
        print('joy')
        return 'An happy tune radiating joy'
    elif predicted_label[0] == 2:
        print('love')
        return 'A heartfelt serenade igniting feelings of love and romance.'
    elif predicted_label[0] == 3:
        print('anger')
        return 'Anger'
    elif predicted_label[0] == 4:
        print('fear')
        return 'A haunting melody evoking shivers of fear and apprehension.'
    elif predicted_label[0] == 5:
        print('surprise')
        return 'A musical surprise that catches you off guard with its twists and turns.'




