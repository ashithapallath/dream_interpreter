# dream_continuation_cpu_fast.py
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras.optimizers import Adam
import pickle
import joblib

def load_data():
    """Optimized data loading with faster pandas settings"""
    try:
        df = pd.read_csv('data/rsos_dream_data.csv', usecols=['text_dream'], engine='c')
        return df['text_dream'].dropna().str.lower().tolist()[:10000]  # Increased subset with better RAM usage
    except FileNotFoundError:
        print("Dataset not found in data/ directory")
        return []

def preprocess_text(texts):
    """Optimized text preprocessing with efficient sequence generation"""
    tokenizer = Tokenizer(num_words=8000, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    
    # Parallel sequence generation
    sequences = tokenizer.texts_to_sequences(texts)
    max_sequence_len = 30  # Reduced sequence length
    
    # Vectorized sequence processing
    input_sequences = []
    for seq in sequences:
        if len(seq) > 1:
            input_sequences.extend([seq[:i+1] for i in range(1, min(len(seq), max_sequence_len))])
    
    input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')
    return input_sequences[:, :-1], input_sequences[:, -1], tokenizer, max_sequence_len

def train_model(X, y, tokenizer, seq_length):
    """Simplified model architecture with faster training"""
    model = Sequential([
        Embedding(input_dim=8001, output_dim=32),  # Fixed input_dim
        GRU(64, return_sequences=False),  # Single GRU layer
        Dense(8001, activation='softmax')  # Pre-calculated vocab size
    ])
    
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate=0.005),  # Higher learning rate
        metrics=['accuracy']
    )
    
    # Single-pass training with full dataset
    model.fit(
        X, y,
        batch_size=128,  # Larger batch size
        epochs=10,       # Reduced epochs
        verbose=1       # Cleaner output
    )
    
    return model

def train_classifier():
    """Optimized classifier training"""
    try:
        df = pd.read_csv('data/rsos_dream_data.csv', engine='c', 
                        usecols=['text_dream', 'emotions_code'])
        df = df.dropna(subset=['text_dream', 'emotions_code'])
        
        # Efficient text processing
        texts = df['text_dream'].str.lower().values
        labels = df['emotions_code'].values
        
        from sklearn.feature_extraction.text import HashingVectorizer  # Memory-friendly
        from sklearn.linear_model import SGDClassifier  # Faster optimizer
        
        vectorizer = HashingVectorizer(n_features=1024, alternate_sign=False)
        X = vectorizer.transform(texts)
        
        clf = SGDClassifier(loss='log_loss', max_iter=200, tol=1e-3)
        clf.fit(X, labels)
        
        joblib.dump((vectorizer, clf), 'classifier_cpu_fast.pkl')
        print("Fast classifier trained in", time.time()-start)
    
    except Exception as e:
        print(f"Training error: {str(e)}")

if __name__ == "__main__":
    # Fast training pipeline
    texts = load_data()
    if texts:
        X, y, tokenizer, seq_length = preprocess_text(texts)
        
        # Save tokenizer first
        with open('tokenizer_cpu_fast.pkl', 'wb') as f:
            pickle.dump((tokenizer, seq_length), f)
            
        # Train and save model
        model = train_model(X, y, tokenizer, seq_length)
        model.save('dream_model_cpu_fast.h5')
        
        # Train classifier
        train_classifier()