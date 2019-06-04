import os
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer 
import string # for punctuation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re



def get_wordnet_pos(word):
    '''Map POS tag to first character lemmatize() accepts'''
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return(tag_dict.get(tag, wordnet.NOUN))  # NOUN is the default


# mystops is a user defined list of stopwords
mystops = ['•' , '–','\uf0a7', '◦', 'i.','\u200b','∗','！']
            
special_stops = ['\u200b']

###################################################
# PRE-PROCESSING FUNCTION

def makeCleanCorpus(dataset,
                    removePunct=True,
                    removeNums=True,
                    lower=True,
                    stops=[],
                    removeStopw=True,
                    lemmatize=False,
                    removeURL=True,
                    makeSentences=False,
                    removeChar = False):
    '''
    The makeCleanCorpus function will look for text files in the directory 
    specified by the dataset. Change this to suit you.
    Input:
        dataset:     Dictionary with the files and its text unprocessed
        removePunct: Should punctuation be removed?
        removeNums:  Should numbers be removed?
        lower:       Should words be converted to lower-case?
        removeStopw: Should stop-words be removed?
        stops:       Apart for the standard English stop-words, the 
                     variable "stops" in the code below allows you to add 
                     your own stop-words
        lemmatize:   Should words be lemmatized? The default is "False"
                     Changing this to true will (really) slow this function down!
        removeURL    Should URLs be removed?
        makeSentence Instead of returning a whole corpus, it returns a list of processed sentences
        removeChar   Remove characters from A to Z that are alone
    Output:
        Dictionary with {name of document : whole corpus or list of sentences -> processed}
    '''
    
    
    ######
    # Objects that needs to be download so that NLTK works for preprocessing
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    ######
    
    # Initialize final dictionary
    clean_files = {}
    
    # Iterates through all the dataset
    for filename, text in dataset.items():
        
        print('Cleaning:', filename)
        
        # Define all the stopwords
        allstopwords = stopwords.words("english") + mystops + stops
                
        # If lemmatize
        if lemmatize:
            # Prepare the Lemmatizer Engine
            lemmatizer = WordNetLemmatizer()
        
        # If remove URL
        if removeURL:
            #text = text.replace("\n"," ")
            # Remove all the URLs with REGEX
            text = re.sub(r"[h|H]ttps?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-zA-Z]{2,6}\b([-a-zA-Z0-9@:%_\+.\\~#?&\/\/=]*)", "", text)
        
        # If remove punctuation and no making sentences
        if (removePunct and not makeSentences):
            # Remove all the punctuations defined in the string library
            text = text.translate(text.maketrans('', '', string.punctuation))
            
        # If remove punctuation except the point (to make sentences)
        if makeSentences:
            # Remove all punctuations except the point, which is needed to create the sentences
            text = text.translate(text.maketrans('', '', string.punctuation.replace(".","")))
        
        # If remove numbers
        if removeNums:
            # Remove all digits
            text = ''.join([x for x in text if not x.isdigit()])
            
        # If convert to lower-case
        if lower:
            # Convert all the text to lower case
            text = text.lower()
        
        if removeChar:
            # Tokenize the whole text
            word_list = nltk.word_tokenize(text)
            # Remove tokens that are a single character
            text = ' '.join([w for w in word_list if (len(w) > 1 or w == '.')])
        
        # If lemmatize words
        if lemmatize:
            # Tokenize the whole text
            word_list = nltk.word_tokenize(text)
            # Apply lemmatization to each token in the text
            text = ' '.join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in word_list])
                
        # If remove stopwords
        if removeStopw:
            # First, replace this special character
            text = text.replace('\u200b',"")
            # Remove Chinese characters
            # text = text.replace(r'[^\x00-\x7F]+', '')
            # Then, include only the words that are not included in the stopwords
            text = ' '.join([word for word in text.split() if word not in allstopwords])
        
        if True:
            # Decoding the text to remove non-ascii characters
            text = text.encode("ascii", "ignore").decode('utf-8')

        # If make sentences
        if makeSentences:
            # Create tokens on a sentence level
            text = nltk.tokenize.sent_tokenize(text)
            # Remove all the points used to separate the sentences
            text = [i.replace(".","") for i in text]
            # Include only sentences with words on it
            text = [x for x in text if x]

        # Add the file to the dictionary, with the code already pre-processed
        clean_files[filename] = text
    
    print ("Done!")
    return(clean_files)


def makeDTM(corpus, tfidf=False):
    '''
        This function will make a Document Term Matrix (DTM) from a corpus 
        passed to it. If tfidf is True, then the DTM returned will have 
        TfIdf values for the terms and not the frequencies.
        Input:
            corpus:     Dictionary with the files and the pre-processed text
            tfidf:      If we want to retrieve only the count of tokens, or the TF-IDF matrix
        Output:
            Pandas Dataframe with the corpus translated into vectors
    '''
    
    if (not tfidf):
        cvec = CountVectorizer()
        smat = cvec.fit_transform(corpus.values())  
        dtm = pd.DataFrame(smat.toarray(), 
                           columns=cvec.get_feature_names(), 
                           index=corpus.keys())
        return(dtm)
    else:
        tvec = TfidfVectorizer()
        tmat = tvec.fit_transform(corpus.values())
        dtm = pd.DataFrame(tmat.toarray(), 
                           columns=tvec.get_feature_names(), 
                           index=corpus.keys())
        return(dtm)

def dictionaryToPandas(dictionary):
    """
        Function that receives a dictionary and it transform it into a Panda DataFrame
        Input:
            Dictionary
        Output:
            DataFrame with index as keys and rows as values
    
    """
    df = pd.DataFrame(None, columns=["text"],index=dictionary.keys())
    for key in dictionary.keys():
        df.loc[key] = [dictionary[key]]
    
    return df   