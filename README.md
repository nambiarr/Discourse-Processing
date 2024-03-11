# Data Cleaning & Visualization Project 
## by: Rtvik Nambiar
Given details about various discussion threads, visualize the:
- most active time period
- most used words
- average word count per post
- average number of posts per user

## Data Cleaning

We will primarily be using the pandas library along with a few others.
Import in the corresponding libraries, as seen in the box below.

After that, locate your file's directory.
They should look something like this: "/Users/.../*Your_File_Name*.csv"



```python
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
```

These are the functions we will use to help us visualize aspects of our data:
- Remove stop words
- NLP function


```python
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

    #After removing the stop words, we now have the words we want to visualize.
    #We can use a frequency distribution for this.
    fdist = FreqDist(filtered_sent)
    fdist.most_common(2)
    fdist.plot(30,cumulative=False)
```

Next, well use the pandas library to read our data files. Then, we'll merge the files based on the column titles: Id & ForumTopicId


```python
#Read in our .csv files.
topics = pd.read_csv(setT)
messages = pd.read_csv(setM) 

#Merge our .csv files to create a master list.
master_list = topics.merge(messages, left_on = 'Id', right_on = 'ForumTopicId')
```

Depending on the needs of our program, we can either visualize our data for all the forums or one specific forum.

For all forums:
- Create a list of unique forum keys in order to perform forum specific actions
- Inside the for loop, create a dataframe for each forum
- Cleaning: remove the NaN values from your 'Message' column

For one forum:
- Create a dataframe for your desired forum
- Cleaning: remove the NaN values from your 'Message' column


```python
#Unique Forum Key - Searches all ForumTopicId's and stores each unique ForumTopicId.
unique_forum_key = master_list['ForumTopicId'].unique()

#This for loop generates a tokenized string of messages from their respective forums.
#for x in unique_forum_key:

#However, we can also comment out the for loop to perform actions on a specific forum.
df = master_list[(master_list.ForumTopicId == 60581)]
df = df[df['Message'].notna()]
```

We can see what the result looks like by printing our dataframe.


```python
print(df.head())
```

             Id_x  ForumId  KernelId  LastForumMessageId  FirstForumMessageId  \
    386449  60581    17686       NaN           1266657.0             353519.0   
    386450  60581    17686       NaN           1266657.0             353519.0   
    386451  60581    17686       NaN           1266657.0             353519.0   
    386452  60581    17686       NaN           1266657.0             353519.0   
    386453  60581    17686       NaN           1266657.0             353519.0   
    
                   CreationDate  Unnamed: 6  \
    386449  07/06/2018 23:27:06         NaN   
    386450  07/06/2018 23:27:06         NaN   
    386451  07/06/2018 23:27:06         NaN   
    386452  07/06/2018 23:27:06         NaN   
    386453  07/06/2018 23:27:06         NaN   
    
                                                        Title  IsSticky    Id_y  \
    386449  Should You Worry That There Aren't Any New Hom...     False  392875   
    386450  Should You Worry That There Aren't Any New Hom...     False  403002   
    386451  Should You Worry That There Aren't Any New Hom...     False  403004   
    386452  Should You Worry That There Aren't Any New Hom...     False  403012   
    386453  Should You Worry That There Aren't Any New Hom...     False  385853   
    
            ForumTopicId  PostUserId             PostDate  ReplyToForumMessageId  \
    386449         60581      356706  09/24/2018 14:30:53                    NaN   
    386450         60581     2332439  10/12/2018 17:54:25               367014.0   
    386451         60581     2332439  10/12/2018 17:57:31               369194.0   
    386452         60581     2332439  10/12/2018 18:03:59                    NaN   
    386453         60581     2236372  09/11/2018 17:30:06                    NaN   
    
                                                      Message  Medal  \
    386449  <h2>What do you think explains this?</h2>\n\n<...    NaN   
    386450  <p>You're right there are no years greater tha...    NaN   
    386451  <p>It's very unlikely that no houses have been...    NaN   
    386452  <p>The dataset is clearly out of date. Because...    NaN   
    386453  <p>I think it's because the data were collecte...    NaN   
    
           MedalAwardDate  
    386449            NaN  
    386450            NaN  
    386451            NaN  
    386452            NaN  
    386453            NaN  


As you can see, the data looks a little nicer. However, our messages have html tags (< p >, < /p >, etc.) at the begining and end. We will continue cleaning the data by removing these html tags.


```python
#In order to use the map() function, 
#we must first create a function for map()'s first argument.
def colander(msg):
    filtered = msg.replace('<p>', '').replace('</p>', '')
    return filtered
    
#Now that we have cleaned our message data, we need to create a string to pass it to our NLP.
mt_string = "".join(map(colander, df['Message']))[1:]
```

After that, we're ready to pass our cleaned data into our functions.


```python
run_nlp(mt_string)
```

    Creating filtered list.



    
![png](img/output_14_1.png)
    



```python

```
