import pandas as pd

from helpers import run_nlp
from helpers import master_list

#Unique Forum Key - Searches all ForumTopicId's and stores each unique ForumTopicId.
unique_forum_key = master_list['ForumTopicId'].unique()

#This for loop generates a tokenized string of messages from their respective forums.
#for x in unique_forum_key:

#However, we can also comment out the for loop to perform actions on a specific forum.
#If we wanted to prompt the user for a Forum Topic Id, we could do that below.
df = master_list[(master_list.ForumTopicId == 60581)]
df = df[df['Message'].notna()]

#Potential Breakpoint: Print new dataframe head
#print(df.head())
    
#In order to use the map() function, 
#we must first create a function for map()'s first argument.
def colander(msg):
    filtered = msg.replace('<p>', '').replace('</p>', '')
    return filtered
    
#Now that we have cleaned our message data, we need to create a string to pass it to our NLP.
mt_string = "".join(map(colander, df['Message']))[1:]
    
run_nlp(mt_string)

