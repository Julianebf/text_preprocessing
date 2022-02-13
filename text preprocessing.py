#!/usr/bin/env python
# coding: utf-8

# # text preprocessing

# In[95]:


# Import libraries
import pandas as pd
import seaborn as sns
import numpy as np
import re
import nltk
import warnings
import random
import string
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from datetime import date
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn import preprocessing 
from scipy import stats
from scipy.stats import norm, skewnorm
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
pd.set_option('display.max_columns', 100)
from nltk import sent_tokenize, word_tokenize
warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# In[96]:


#Load data
df_customers = pd.read_csv('C:\Data\olist_customers_dataset.csv', delimiter = ',',quotechar="'")
df_order_reviews = pd.read_csv('C:\Data\olist_order_reviews_dataset.csv', delimiter = ',', encoding = 'utf-8')
df_geolocations = pd.read_csv('C:\Data\olist_geolocation_dataset.csv', delimiter = ',' )
df_order_payments = pd.read_csv('C:\Data\olist_order_payments_dataset.csv', delimiter = ',', quotechar="'")
df_categorys = pd.read_csv('C:\Data\product_category_name_translation.csv', sep = ',')
df_sellers = pd.read_csv('C:\Data\olist_sellers_dataset.csv', delimiter = ',')
df_products = pd.read_csv('C:\Data\olist_products_dataset.csv', delimiter = ',', quotechar="'")
df_orders_items = pd.read_csv('C:\Data\olist_order_items_dataset.csv', delimiter = ',')
df_orders = pd.read_csv('C:\Data\olist_orders_dataset.csv', delimiter = ',', quotechar="'", encoding = 'utf-8')


# ##### text variable
# 

# In[132]:


text_copy = main_df.copy()


# In[133]:


text_copy.isnull().sum()


# In[134]:


text_copy.drop_duplicates(keep='first', inplace=True) 


# In[135]:


text_copy.head()


# In[136]:


text_copy['Tag'] = text_copy['message'].isnull()*1
text_copy


# In[137]:


text_copy = text_copy[pd.notnull(text_copy['message'])]
text_copy


# In[138]:


pd.options.display.max_colwidth = 300
msg_df = text_copy[['message']]
msg_df


# In[139]:


text_copy['message'] = text_copy['message'].map(lambda x: re.sub('[,\.!?></]', '', x))
text_copy['message'] = text_copy['message'].map(lambda x: x.lower())


# In[140]:


text_copy[['message']]


# In[141]:


text_copy['message'] = text_copy['message'].replace('0','',regex=True).astype(str)
text_copy['message'] = text_copy['message'].replace('1','',regex=True).astype(str)
text_copy['message'] = text_copy['message'].replace('2','',regex=True).astype(str)
text_copy['message'] = text_copy['message'].replace('3','',regex=True).astype(str)
text_copy['message'] = text_copy['message'].replace('4','',regex=True).astype(str)
text_copy['message'] = text_copy['message'].replace('5','',regex=True).astype(str)
text_copy['message'] = text_copy['message'].replace('6','',regex=True).astype(str)
text_copy['message'] = text_copy['message'].replace('7','',regex=True).astype(str)
text_copy['message'] = text_copy['message'].replace('8','',regex=True).astype(str)
text_copy['message'] = text_copy['message'].replace('9','',regex=True).astype(str)
text_copy['message'] = text_copy['message'].replace('borrower added on ','',regex=True).astype(str)


# In[143]:


def identify_tokens(row):
    message = row['message']
    tokens = nltk.word_tokenize(message)
    token_words = [w for w in tokens if w.isalpha()]
    return token_words


# In[144]:


text_copy['message_clear'] = text_copy.apply(identify_tokens, axis=1)


# In[145]:


text_copy[['message','message_clear']]


# In[146]:


stemming = PorterStemmer()

my_list = []


# In[147]:


def stem_list(row):
    text_copy = row['message_clear']
    stemmed_list = [stemming.stem(message_clear) for message_clear in text_copy]
    return (stemmed_list)


text_copy['message_stemmed'] = text_copy.apply(stem_list, axis=1)


# In[148]:


text_copy[['message','message_clear','message_stemmed']]


# In[149]:


stops = stopwords.words('portuguese')
def remove_stops(row):
    my_list = row['message_stemmed']
    meaningful_message = [w for w in my_list if not w in stops]
    return (meaningful_message)

text_copy['stem_meaningful'] = text_copy.apply(remove_stops, axis=1)


# In[150]:


text_copy[['message','message_clear','message_stemmed','stem_meaningful']]


# In[151]:


def rejoin_words(row):
    text_copy = row['stem_meaningful']
    joined_message= ( " ".join(text_copy))
    return joined_message

text_copy['processed'] = text_copy.apply(rejoin_words, axis=1)
text_copy

data_lemmatized = text_copy.copy()


# In[152]:


cols_to_drop = ['message', 'message_clear', 'message_stemmed', 'stem_meaningful']
text_copy.drop(cols_to_drop, inplace=True, axis=1)
text_copy ['processed']

