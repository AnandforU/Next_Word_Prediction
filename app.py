import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the model
model = load_model('next_word_lstm.h5')

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

def predict_next_word(model, tokenizer, text, max_len, num_words=10):
    predicted_text = text
    for _ in range(num_words):
        token_text = tokenizer.texts_to_sequences([predicted_text])
        padded_token_text = pad_sequences(token_text, maxlen=max_len-1, padding='pre')
        prob = model.predict(padded_token_text)
        word_index = np.argmax(prob, axis=-1)[0]
        
        # Find the word corresponding to the index
        word = None
        for w, i in tokenizer.word_index.items():
            if i == word_index:
                word = w
                break
        
        if word is None:
            break
        
        predicted_text = f"{predicted_text} {word}"
    
    return predicted_text

st.title("Next Word Prediction")
text = st.text_input("Enter the sequence of words")

if st.button("Predict Next Word"):
    if text:
        max_len = model.input_shape[1] + 1
        next_word = predict_next_word(model, tokenizer, text, max_len, num_words=10)
        st.write(f"Predicted sequence: {next_word}")
    else:
        st.write("Please enter a sequence of words.")
