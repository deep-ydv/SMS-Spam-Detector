# app.py
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

app = Flask(__name__)

# Text preprocessing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenization and removing stopwords
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return ' '.join(words)

# Load and preprocess the data
def prepare_data():
    # Read CSV file with proper encoding and handle unnamed columns
    data = pd.read_csv('spam.csv', encoding='latin-1', usecols=[0, 1])
    
    # Rename columns to ensure they match our expected names
    data.columns = ['v1', 'v2']
    
    # Preprocess the messages
    data['processed_text'] = data['v2'].apply(preprocess_text)
    
    return data
# Train the model
def train_model(data):
    # Create and fit the vectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data['processed_text'])
    
    # Convert labels to numerical values
    y = (data['v1'] == 'spam').astype(int)
    
    # Train the model
    model = MultinomialNB()
    model.fit(X, y)
    
    # Save the model and vectorizer
    with open('model.pkl', 'wb') as f:
        pickle.dump((model, vectorizer), f)

# Initialize and train the model
data = prepare_data()
train_model(data)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the message from the request
    message = request.json['message']
    
    # Load the model and vectorizer
    with open('model.pkl', 'rb') as f:
        model, vectorizer = pickle.load(f)
    
    # Preprocess the message
    processed_message = preprocess_text(message)
    
    # Transform the message
    message_vector = vectorizer.transform([processed_message])
    
    # Make prediction
    prediction = model.predict(message_vector)[0]
    probability = model.predict_proba(message_vector)[0][1]
    
    # Return the result
    result = {
        'is_spam': bool(prediction),
        'probability': float(probability),
        'message': message
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)