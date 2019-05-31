# -*- coding: utf-8 -*-

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

###########################################################
###########################################################
###########################################################


##### TODO

# - Remove URL
# - Function that receives a text and generate a corpus per sentence / parragraph.
# - Review Lemmatization, seems not to be working
########

# a function to POS-tag each word for lemmatization


def get_wordnet_pos(word):
    '''Map POS tag to first character lemmatize() accepts'''
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return(tag_dict.get(tag, wordnet.NOUN))  # NOUN is the default


# mystops is a user defined list of stopwords
mystops = ['mr', 'said', 'sir', 'upon', 'mrs', 
                                    'replied', 'man', 'one', 'little', 
                                    'say', 'dont', 'old', 'gentleman',
                                    'time', 'two', 'never', 'see', 'door',
                                    'dear', 'well', 'now', 'will', 'dear',
                                    'know', 'head', 'come', 'much', 'hand',
                                    'o', 'inquired', 'room', 'think',
                                    'way', 'away', 'great', 'good', 'gentleman', 
                                    'lady', 'long', 'first', 'made', 
                                    'back', 'another', 'can', 'take', 'must',
                                    'just', 'ever', 'face', 'nothing', 'without',
                                    'ever', 'shall', 'took', 'look', 'friend',
                                    'oh', 'yes', 'many','last', 'might', 'go',
                                    'may', 'looking', 'rather', 'got', 'place',
                                    'mind', 'right', 'house', 'three','every', 'day',
                                    'put', 'thats', 'quite',
                                    'call', 'could', 'even', 'eye', 'get', 'give', 'let', 'make', 'open',
           'reply', 'turn', 'would', '•' , '–','\uf0a7', '◦', 'i.']
            


def makeCleanCorpus(dataset,
                    removePunct=True,
                    removeNums=True,
                    lower=True,
                    stops=[],
                    removeStopw=True,
                    lemmatize=False,
                    removeURL=True,
                    makeSentences=False):
    '''
    The makeCleanCorpus function will look for text files in the directory 
    specified by abspath. Change this to suit you.
    Input:
        dataset:     Dictionary with the files and its text
        removePunct: Should punctuation be removed?
        removeNums:  Should numbers be removed?
        lower:       Should words be converted to lower-case?
        removeStopw: Should stop-words be removed?
                     Apart for the standard English stop-words, the 
                     variable "mystops" in the code below allows you to add 
                     your own stop-words
        lemmatize:   Should words be lemmatized? The default is "False"
                     Changing this to true will (really) slow this function down!
    '''
    ######
    nltk.download('averaged_perceptron_tagger')
    ######
    
    
    clean_files = {}
        
    for filename, text in dataset.items():
        
        print('Cleaning:', filename)
        
        StopWords = stopwords.words("english") + mystops
        
        if lemmatize:
            lemmatizer = WordNetLemmatizer()
        
        if removeURL:
            #text = text.replace("\n"," ")
            text = re.sub(r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.\\~#?&\/\/=\s]*)", "", text)
        
        # remove punctuation
        # use the three argumnt form of text.maketrans()
        if (removePunct and not makeSentences):
            text = text.translate(text.maketrans('', '', string.punctuation))
            
        # remove punctuation except the point (to make sentences)
        if makeSentences:
            text = text.translate(text.maketrans('', '', string.punctuation.replace(".","")))
        
        # remove numbers
        if removeNums:
            text = ''.join([x for x in text if not x.isdigit()])
            
        # convert to lower-case
        if lower:
            text = text.lower()
                       
        # lemmatize words
        if lemmatize:
            word_list = nltk.word_tokenize(text)    
            text = ' '.join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in word_list])
                
        # remove stopwords
        if removeStopw:
            text = ' '.join([word for word in text.split() if word not in StopWords])
             
        if makeSentences:
            text = nltk.tokenize.sent_tokenize(text)
            text = [i.replace(".","") for i in text]
            text = [x for x in text if x]

        
        clean_files[filename] = text
    
    print ("Done!")
    return(clean_files)
# end of function makeCleanCorpus

def makeDTM(corpus, tfidf=False):
    '''
        This function will make a Document Term Matrix (DTM) from a corpus 
        passed to it. If tfidf is True, then the DTM returned will have 
        TfIdf values for the terms and not the frequencies.
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

# end of function makeDTM
        
#################################

# make a clean corpus and then return the cleaned up corpus   
# for a list of sentences


################################################################
################################################################

# this works, but is still experimental

# controlling sparsity

# LOGIC: replace each non-zero frequency in the DTM with a 1
# Then, the sum of col is the number of times the term occurs in the corpus.
# The sparsity is this col sum divided by the number of docs 

# percent is a number between 1 and 100
def controlSparsity(percent):
    
    dtm.iloc[:,1]
    
    dtm.iloc[1]  # some values > 1
    occ_dtm = dtm.copy(deep=True)
    occ_dtm.iloc[1]
    occ_dtm[occ_dtm != 0] = 1
    occ_dtm.iloc[1]  # all non-zero val are 1
    dtm.iloc[1]  # has not changed - that's what a deep copy does
    
    # now use the occurence matrix occ_dtm to calculate sparsesness
    
    occ_col_sums = occ_dtm.sum()  # the no of docs in which each term appears
    
    occ_col_sums[occ_col_sums==5]
    
    # calculate the sparsity
    num_docs = len(dtm.index)
    num_docs
    sparsity = occ_col_sums/num_docs
    sparsity
    len(sparsity)
    
    # got the sparsity
    # now remove terms with sparsity less than specified level
    sparsity_cutoff = percent/100
    
    
    sp = sparsity[sparsity>sparsity_cutoff]
    len(sp)
    sp
    
    
    dtm.index
    dtm.columns
    
    sp.index
    
    # for sparsity > sparsity_cutoff, keep these indices:
    keep_indices = dtm.columns.difference(dtm.columns.difference(sp.index))
    keep_indices
    
    # and here is the sparse DTM!
    dtm.sparse95 = dtm.loc[:, keep_indices]
    # the above throws a warning (?) the first time you run it, but it works :)
    
    
    return(dtm.sparse95)


#############################################