from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups


nltk.download('stopwords')

app = Flask(__name__)

newsgroups = fetch_20newsgroups(subset='all')

vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'), max_features=1000,ngram_range=(1, 2),min_df=4,max_df=0.8)

term_doc_mat = vectorizer.fit_transform(newsgroups.data)

svd = TruncatedSVD(n_components=100) 

svd_marix = svd.fit_transform(term_doc_mat)



def search_engine(query):
    """
    Function to search for top 5 similar documents given a query
    Input: query (str)
    Output: documents (list), similarities (list), indices (list)
    """
    print("meow", query)
    tf = vectorizer.transform([query])
    cosine_sim_tfidf = cosine_similarity(tf, term_doc_mat)

    similarity_scores_lsa = cosine_sim_tfidf[0]

    results_lsa = [(index, similarity_scores_lsa[index], newsgroups.data[index]) 
               for index in np.argsort(similarity_scores_lsa)[::-1]]
    
    indices, similarities, documents = zip(*results_lsa[:5])
    indices = [int(idx) for idx in indices]
    similarities = [float(sim) for sim in similarities]
    return list(documents), list(similarities), list(indices)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    documents, similarities, indices = search_engine(query)
    return jsonify({'documents': documents, 'similarities': similarities, 'indices': indices}) 

if __name__ == '__main__':
    app.run(port=3000, debug=True)
