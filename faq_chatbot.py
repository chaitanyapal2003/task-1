from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import string
from flask import Flask, request, render_template
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Sample FAQs
faq_data = [
    {"question": "What is your return policy?", "answer": "Our return policy allows returns within 30 days."},
    {"question": "How can I track my order?", "answer": "You can track your order using the tracking link sent to your email."},
    {"question": "Do you offer customer support?", "answer": "Yes, we offer 24/7 customer support through chat and email."},
    {"question": "What payment methods are accepted?", "answer": "We accept Visa, Mastercard, PayPal, and UPI."},
    {"question": "Can I cancel my order?", "answer": "Yes, you can cancel your order before it is shipped."}
]

stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words and w not in string.punctuation]
    return ' '.join(tokens)

# Prepare corpus
questions = [preprocess(faq["question"]) for faq in faq_data]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(questions)

# Chatbot response logic
def get_response(user_input):
    user_input = preprocess(user_input)
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, tfidf_matrix)
    idx = similarity.argmax()

    if similarity[0][idx] < 0.3:
        return "Sorry, I didn't understand that. Please try again."
    return faq_data[idx]["answer"]
