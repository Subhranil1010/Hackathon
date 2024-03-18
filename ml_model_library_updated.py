# File: ml_model_library.py

import ssl
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

# Create an unverified SSL context
ssl._create_default_https_context = ssl._create_unverified_context
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
import nltk
# nltk.download('stopwords')
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Simple text preprocessing
def preprocess_text(text, use_stemming=True):
    text = text.lower()  # Lowercase text
    text = re.sub(r'\W+', ' ', text)  # Remove all non-word characters

    
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    if use_stemming:
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

def evaluate(context, documents, category, threshold, no_of_matches):
    """
    Evaluate the relevance of documents to the given context using TF-IDF and cosine similarity.
    """
    context_preprocessed = preprocess_text(context)
    documents_preprocessed = [preprocess_text(doc['content']) for doc in documents if 'content' in doc and preprocess_text(doc['content']).strip()]

    # Check if there are any documents left after preprocessing
    if not documents_preprocessed:
        return {
            "Status": "Error",
            "Message": "No valid documents found after preprocessing.",
            "Metadata": {"confidenceScore": threshold},
            "Results": []
        }
    
    # Model initialization
    model = Doc2Vec(vector_size = 50,
    min_count = 5,
    epochs = 100,
    alpha = 0.001
    )
    
    # Train the model
    for epoch in range(model.epochs):
    print(f"Training epoch {epoch+1}/{model.epochs}")
    model.train(documents_preprocessed, 
                total_examples=model.corpus_count, 
                epochs=model.epochs)

    model.save('cv_job_maching.model')
    print("Model saved")
    
    
    # Model evaluation
    model = Doc2Vec.load('cv_job_maching.model')
    v1 = model.infer_vector(context_preprocessed.split())
    v2 = model.infer_vector(documents_preprocessed.split())
    cosine_similarities = 100*(np.dot(np.array(v1), np.array(v2))) / (norm(np.array(v1)) * norm(np.array(v2)))
    
    #cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    print(round(similarity, 2))
        # Find top N matching documents
    top_indices = cosine_similarities.argsort()[-no_of_matches:][::-1]
    top_scores = cosine_similarities[top_indices]

    results = [{
        "Id": documents[idx]['id'],
        "Score": round(score, 2),
        "Path": documents[idx]['path']
    } for idx, score in zip(top_indices, top_scores) if score >= threshold]

    # Check if there are any matching documents
    if not results:
        return {
            "Status": "Error",
            "Message": "No documents meet the threshold criteria.",
            "Metadata": {"confidenceScore": threshold},
            "Results": []
        }

    return {
        "Status": "Success",
        "Count": len(results),
        "Metadata": {"confidenceScore": threshold},
        "Results": results
    }



