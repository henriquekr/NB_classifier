# Title: NB Classifier for e-commerce detection
# Language: Python (3.x)
# Author: Henrique Kokron Rodrigues    
#                          
# Description: This script takes as input a list of urls with labels indicating whether
#              it's an e-commerce or not. It then scrapes the websites to get the visible text
#              from the HTML and, based on the text data gathered, uses a bag of words model
#              to build a Naive Bayes classifier to check if a website has a good chance of being an e-commerce.
#
# References: 1. https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a
#             2. http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
#             3. https://stackoverflow.com/questions/1936466/beautifulsoup-grab-visible-webpage-text

               
##################
##### IMPORTS #####
###################

import pandas as pd
import numpy as np
from langdetect import detect
import requests
from bs4.element import Comment
from bs4 import BeautifulSoup
import time
import random

#####################
##### FUNCTIONS #####
#####################

def clean_URL(url):
	if url[:4] == 'www.':
		url = url[4:]
	elif url[:11] == 'http://www.':
		url = url[11:]
	elif url[:12] == 'https://www.':
		url = url[12:]
	elif url[:7] == 'http://':
		url = url[7:]
	elif url[:8] == 'https://':
		url = url[8:]
	if url[-1:] == '/':
		url = url[:-1]
	return url

def clean_URL2(url):
    cleanurl = url
    slash = cleanurl.find('/')
    if  slash != -1:
        cleanurl = url[:slash]
    question = cleanurl.find('?')
    if question != -1:
        cleanurl = url[:question]
    return cleanurl

def tag_visible(element): #texts from the GetVisibleText functions are passed as 'element' inputs
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def get_visible_text(oSoup):
    texts = oSoup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)  
    return u" ".join(t.strip() for t in visible_texts)   

# Returns a hash with domains as keys and website information as values
def website_scrape(domain,dhash):
    domain_hash = dhash,copy()
    domain_hash[domain]={}
    url = 'http://www.'+ domain
    oResponse = requests.get(url,timeout=10)
    iResp_Status = oResponse.status_code
    if iResp_Status>=400:
        domain_hash[domain]['Website_error'] = 'access error - 400 or 500'
        domain_hash[domain]['Final_url_domain']= np.nan
        domain_hash[domain]['Redirect'] = np.nan
        domain_hash[domain]['Website_language'] = np.nan
        domain_hash[domain]['Website_text']= np.nan
        domain_hash[domain]['Website_text_length'] = np.nan
    else:
        domain_hash[domain]['Website_error'] = 'no'
        sFinalUrl = oResponse.url
        sFinalUrlDomain = clean_URL(sFinalUrl)
        sFinalUrlDomain = clean_URL2(sFinalUrlDomain)
        domain_hash[domain]['Final_url_domain']= sFinalUrlDomain
        print('URL clean OK:' + sFinalUrlDomain)
        bRedirect = False
        if iResp_Status>=300:
            bRedirect = True
        domain_hash[domain]['Redirect'] = bRedirect
        #get visible text
        sPageText = oResponse.text 
        oSoup = BeautifulSoup(sPageText,'lxml')
        sVisibleText = get_visible_text(oSoup)
        domain_hash[domain]['Website_text'] = sVisibleText
        domain_hash[domain]['Website_text_length'] = len(sVisibleText)
        if len(sVisibleText) != 0:
            sLanguage = detect(sVisibleText)
        else:
            sLanguage = 'no visible text'
        domain_hash[domain]['Website_language'] = sLanguage
    return domain_hash

################
##### MAIN #####
################

### READ INPUT DATA ### 
# input file has two columns, first one with the domain and second with the label 
dfWebsitelist = pd.read_excel('domains_database.xlsx',header_col = 0)
dfWebsitelist['domain_clean'] = dfWebsitelist['url'].apply(lambda x: clean_URL(x))
dfWebsitelist['domain_clean'] = dfWebsitelist['domain_clean'].apply(lambda x: clean_URL2(x))
dfWebitelsit.drop_duplicates(subset =['domain_clean'], keep='first', inplace=True)
dfWebsitelist.set_index('domain_clean',inplace=True)
aDomainlist = list(dfWebsitelist.index)

### LOOP TO GET WEBSITE INFO ###
# all website info will be stored in a dictionary with each domain as a key.
Domains_hash = {}
start = time.clock() #check how much times it takes to go through all the websites
for domain in aDomainlist:
    print("Accessing: " + domain)
    iTry = 0
    success = False
    while (iTry<5 and success == False):
        try:
            Domains_hash = website_scrape(domain,Domains_hash)
            success = True
        except:
           #print('ERROR')
           time.sleep(5)
           iTry += 1
        if success == False:
           Domains_hash[domain] = {} 
           Domains_hash[domain]['Website_error'] = 'IP error'
           Domains_hash[domain]['Final_url_domain']= np.nan
           Domains_hash[domain]['Redirect'] = np.nan
           Domains_hash[domain]['Website_language'] = np.nan
           Domains_hash[domain]['Website_text']= np.nan        
           Domains_hash[domain]['Website_text_length']= np.nan   

finish = time.clock()
run_time_minutes = (finish-start)/60
print('RUNTIME TOTAL = ' + str(run_time_minutes))

### PREPARE DATA FOR NB CLASSIFIER ###
# get only websites with no access error 
hDomains_cleaned = Domains_hash.copy()
for domain in Domains_hash.keys():
    if Domains_hash[domain]['Website_error']!='no':
        del hDomains_cleaned[domain]

# shuffle elements to split into train and test set
aKeys = list(hDomains_cleaned.keys())
random.shuffle(aKeys)
iSize = len(aKeys)
iSplit = round(iSize*0.7) #70% to train set and 30% to test set
aTrain = aKeys[:iSplit] 
aTest = aKeys[iSplit:]

# The lists marked with _data contain the text data that feeds the classifier
# while the _target lists contain the labels
aTrain_data = []       
aTrain_target = []
for domain in aTrain:
    aTrain_data.append(hDomains_cleaned[domain]['Website_text'])
    aTrain_target.append(dfWebsitelist.loc[domain,'ecommerce'])
aTest_data = []       
aTest_target = []  
for domain in aTest:
    aTest_data.append(hDomains_cleaned[domain]['Website_text'])
    aTest_target.append(dfWebsitelist.loc[domain,'ecommerce'])

### BUILD CLASSIFIER ###
# import list of portuguese stop words; these should be ignored by the model 
sStop_words = open('stop_words_pt.txt').read()
aStop_words = sStop_words.split('\n')

# creates a matrix with word counts for each element in the training set.
# every word from every text is viewed as a feature.
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(stop_words = aStop_words)
X_train_counts = count_vect.fit_transform(aTrain_data)
X_train_counts.shape
# transforms the wordcount into a frequency measure for each word.
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

# creates simiilar matrix for the test set, but taking into account
# the words found on the training set
X_test_counts = count_vect.transform(aTest_data)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)
X_test_tfidf.shape

# creates a classifier using the Multinomial Naive Bayes model
# according to the sklearn documentation, this variant of the NB method 
# "is suitable for classification with discrete features (e.g., word counts for text classification)."
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, aTrain_target)

# Assess the performance of the model on the test set
predicted = clf.predict(X_test_tfidf)
predicted_proba = clf.predict_proba(X_test_tfidf)  
clf.score(X_test_tfidf,aTest_target)

# creates a Confusion Matrix to check how many false 
# positives and false nagatives came up
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(aTest_target, predicted)

### FINAL WORDS ###
# This script was adapted from a script developped by me at my current job. A test 
# was done for US websites, whith a little over 1000 URLs and the accuracy was abo-
# ve 90%. The high sucess rate could also be explained by the fact the ecommerce 
# websites share a similar structure, listing categories, indicating prices and so 
# on, making it easier to distinguish them from blogs or news. An interesting deve-
# lopment would be to try and categorize the websites by vertical (e.g. fashion, 
# sporting goods and travel). 


