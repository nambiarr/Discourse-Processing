import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import matplotlib as plt

#Locate .csv files.
#NOTE: We are using test data
setT = '/Users/Rtvik/NLP/input/Topics.csv' 
setM = '/Users/Rtvik/NLP/input/Messages.csv'

#Read in our .csv files.
topics = pd.read_csv(setT)
messages = pd.read_csv(setM) 

#Merge our .csv files to create a master list.
master_list = topics.merge(messages, left_on = 'Id', right_on = 'ForumTopicId')

#Converting our new dataframe into a .csv file
master_list.to_csv('output/master_list.csv', encoding='utf-8')

def remove_stop_words(forum_messages):
    #run_nlp takes a string of sentences that we need to convert into token words.
    #This is so that we are able to analyze each word.
    tokenized_words = word_tokenize(forum_messages)

    #After breaking down each sentence, we need to get rid of irrelevant words.
    #stop_words are the words we wish to remove.
    stop_words = set(stopwords.words("english"))
    
    #Removing StopWords
    filtered_sent = []
    for x in tokenized_words:
        if x not in stop_words:
            filtered_sent.append(x)
    
    other_characters = {".", " ,", ", ", ",", "<", ">", "(", ")", '"', ":", "?", "I", "the"}
    for x in tokenized_words:
        if x not in other_characters:
            filtered_sent.append(x)
    
    return filtered_sent

def run_nlp(forum_messages):    
    #Removing StopWords
    filtered_sent = remove_stop_words(forum_messages)

    #print(filtered_sent)
    #print(master_list.head(20))
    #After removing the stop words, we now have the words we want to visualize.
    #We can use a frequency distribution for this.
    fdist = FreqDist(filtered_sent)
    fdist.most_common(2)
    fdist.plot(30,cumulative=False)









#def activity_hotspot(forum_messages):
#def activity_hotspot(forum_messages):
#def activity_hotspot(forum_messages):
  
  
    

