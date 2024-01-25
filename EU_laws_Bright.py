#!/usr/bin/env python
# coding: utf-8

# In[35]:


import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')



# In[36]:


eu_regulation_url = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32019R0947&from=EN"



# In[37]:


def convert_text(url):                               #this converts the url to actual text
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.get_text()


# In[26]:


eu_regulation_text = convert_text(eu_regulation_url)
eu_regulation_text


# In[61]:


def analyze_sentences(text):                 #by tokenizing the senteces we separete them one by one and we get the number of words per sentence
    sentences = sent_tokenize(text)
    sentence_lengths = [len(word_tokenize(sentence)) for sentence in sentences]
    num_sentences = len(sentences)
    length_distribution = nltk.FreqDist(sentence_lengths)
    return num_sentences, length_distribution


# In[62]:


analyze_sentences_laws = analyze_sentences(eu_regulation_text)
analyze_sentences_laws


# In[41]:


def calculate_similarity(sentences):        #we get the similarity between sentences compared to the respectively Echelor Form matrix position.
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    similarity_matrix = (tfidf_matrix * tfidf_matrix.T).A
    return similarity_matrix


# In[49]:


sentences = sent_tokenize(eu_regulation_text)
calculate_similarity(sentences)


# In[50]:


def plot_similarity_graph(similarity_matrix):               #the similarity graph
    plt.imshow(similarity_matrix, cmap='viridis', interpolation='nearest')
    plt.title('Sentence Similarity Graph')
    plt.colorbar()
    plt.show()


# In[52]:


plot_similarity_graph(calculate_similarity(sentences))


# In[59]:


def get_keywords(text, num_keywords=10):        #we get the keywords by Term Frecuency in the document 
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    sorted_indices = tfidf_matrix.sum(axis=0).A1.argsort()[::-1]
    top_keywords = [feature_names[idx] for idx in sorted_indices[:num_keywords]]
    return top_keywords


# In[60]:


get_keywords(eu_regulation_text)


# In[47]:




