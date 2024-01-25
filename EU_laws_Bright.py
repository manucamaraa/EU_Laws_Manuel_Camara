
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')



eu_regulation_url = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32019R0947&from=EN"



def convert_text(url):                               #this converts the url to actual text
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.get_text()




eu_regulation_text = convert_text(eu_regulation_url)
eu_regulation_text



## Count the number of sentences 
## Calculate the length of sentences (as in words per sentence) and then show the overall distribution of sentence length for the document.

def analyze_sentences(text):                 #by tokenizing the senteces we separete them one by one and we get the number of words per sentence
    sentences = sent_tokenize(text)
    sentence_lengths = [len(word_tokenize(sentence)) for sentence in sentences]
    num_sentences = len(sentences)
    length_distribution = nltk.FreqDist(sentence_lengths)
    return num_sentences, length_distribution


analyze_sentences_laws = analyze_sentences(eu_regulation_text)
analyze_sentences_laws



## Similarity and graph

def calculate_similarity(sentences):        #we get the similarity between sentences compared to the respectively Echelor Form matrix position.
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    similarity_matrix = (tfidf_matrix * tfidf_matrix.T).A
    return similarity_matrix


sentences = sent_tokenize(eu_regulation_text)
calculate_similarity(sentences)


def plot_similarity_graph(similarity_matrix):               #the similarity graph
    plt.imshow(similarity_matrix, cmap='viridis', interpolation='nearest')
    plt.title('Sentence Similarity Graph')
    plt.colorbar()
    plt.show()


plot_similarity_graph(calculate_similarity(sentences))



## BONUS PART: get the keywords 
def get_keywords(text, num_keywords=10):        #we get the keywords by Term Frecuency in the document 
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    sorted_indices = tfidf_matrix.sum(axis=0).A1.argsort()[::-1]
    top_keywords = [feature_names[idx] for idx in sorted_indices[:num_keywords]]
    return top_keywords


get_keywords(eu_regulation_text)







