import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import joblib
import sqlite3
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
MODEL_PATH = "dream_model_cpu_fast.h5"
TOKENIZER_PATH = "tokenizer_cpu_fast.pkl"
CLASSIFIER_PATH = "classifier_cpu_fast.pkl"
DB_PATH = "dreams.db"

# Initialize Database
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS dreams 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                 user_input TEXT, 
                 generated_dream TEXT, 
                 interpretation TEXT)''')
    conn.commit()
    conn.close()

init_db()

# Load components with error handling
@st.cache_resource
def load_components():
    try:
        model = load_model(MODEL_PATH, compile=False)
        with open(TOKENIZER_PATH, 'rb') as f:
            tokenizer, seq_length = pickle.load(f)
        vectorizer, clf = joblib.load(CLASSIFIER_PATH)
        return model, tokenizer, seq_length, vectorizer, clf
    except FileNotFoundError as e:
        st.error(f"❌ Missing file: {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading components: {str(e)}")
        st.stop()

# Dream Continuation Function
def generate_dream(model, tokenizer, seq_length, seed_text, num_words=100, temperature=0.8):
    """Generates a dream continuation using AI."""
    seed_text = seed_text.lower().replace('.', ' . ').replace(',', ' , ')
    generated = seed_text

    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([generated])[0][-seq_length+1:]
        token_list = pad_sequences([token_list], maxlen=seq_length-1, padding='pre')

        predictions = model.predict(token_list, verbose=0)[0]
        exp_preds = np.exp(np.log(predictions) / temperature)
        preds = exp_preds / np.sum(exp_preds)
        predicted_index = np.random.choice(len(preds), p=preds)

        next_word = tokenizer.index_word.get(predicted_index, '')

        if any([
            len(generated.split()) >= 150,
            generated.count('.') >= 3,
            next_word in ['end', 'finished', 'woke']
        ]):
            break

        generated += " " + next_word

    sentences = [s.strip().capitalize() + '.' for s in generated.split('. ') if s.strip()]
    return ' '.join(sentences)

# Dream Interpretation Function
def generate_interpretation(text):
    """Returns an AI-generated interpretation based on dream themes."""
    keywords = {
        'family': ['mother', 'father', 'sister', 'brother', 'family'],
        'anxiety': ['alone', 'testing', 'pressure', 'didnt let'],
        'nature': ['green', 'forest', 'water', 'lake', 'sky']
    }

    themes = [theme for theme, words in keywords.items() if any(word in text.lower() for word in words)]
    
    interpretations = {
        'family': "This dream suggests unresolved family dynamics or relationship considerations.",
        'anxiety': "The dream indicates underlying stress or performance anxiety.",
        'nature': "May represent growth, freedom, or a desire to connect with nature."
    }

    return '\n\n'.join([interpretations[theme] for theme in themes if theme in interpretations]) or "This dream contains complex symbolism worth personal reflection."

# Store dream analysis in database
def save_to_db(user_input, generated_dream, interpretation):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO dreams (user_input, generated_dream, interpretation) VALUES (?, ?, ?)", 
              (user_input, generated_dream, interpretation))
    conn.commit()
    conn.close()

# Retrieve similar dreams using TF-IDF
def find_similar_dream(user_input):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT user_input, generated_dream, interpretation FROM dreams")
    past_dreams = c.fetchall()
    conn.close()

    if not past_dreams:
        return None, None, None

    dream_texts = [d[0] for d in past_dreams] + [user_input]
    vectorizer = TfidfVectorizer().fit_transform(dream_texts)
    similarities = cosine_similarity(vectorizer[-1], vectorizer[:-1]).flatten()

    best_match_idx = np.argmax(similarities)
    if similarities[best_match_idx] > 0.75:  # Similarity threshold
        return past_dreams[best_match_idx]
    return None, None, None

# Streamlit UI Configuration
st.set_page_config(page_title="Dream Interpreter AI", page_icon="🌙", layout="centered")

st.markdown("## 🌌 Dream Interpreter AI")
st.markdown("Describe your dream to receive an AI-generated continuation and interpretation.")

# Input Section
user_input = st.text_area("**Describe your dream:**", height=150, placeholder="Example: I was running through a forest but couldn't find the path...")

# Load components once
model, tokenizer, seq_length, _, _ = load_components()

# Analyze Button
if st.button("✨ Analyze Dream", use_container_width=True):
    if not user_input.strip():
        st.error("Please describe your dream fragment to begin")
        st.stop()
    
    cleaned_input = re.sub(r'[^a-zA-Z0-9\s.,]', '', user_input)
    
    if len(cleaned_input.split()) < 5:
        st.warning("Please provide more details (at least 5 words) for better analysis")
        st.stop()
    
    # Check for similar dreams in the database
    similar_dream, similar_continuation, similar_interpretation = find_similar_dream(cleaned_input)

    if similar_dream:
        st.markdown("### 🔁 Found a Similar Dream")
        st.markdown(f"> {similar_dream}")
        st.markdown("### 🌙 Dream Continuation")
        st.markdown(f"> {similar_continuation}")
        st.markdown("### 🔮 AI Interpretation")
        st.markdown(f"> {similar_interpretation}")
    else:
        with st.spinner("🌠 Analyzing dream patterns..."):
            try:
                generated_dream = generate_dream(model, tokenizer, seq_length, cleaned_input)
                interpretation = generate_interpretation(generated_dream)

                # Save to database
                save_to_db(cleaned_input, generated_dream, interpretation)

                st.markdown("### 🌙 Dream Continuation")
                st.markdown(f"> {generated_dream}")

                st.markdown("### 🔮 AI Interpretation")
                st.markdown(f"> {interpretation}")
            
            except Exception as e:
                st.error(f"Error processing dream: {str(e)}")

st.markdown("💡 *Disclaimer: This AI-generated dream analysis is for entertainment purposes only.*")
