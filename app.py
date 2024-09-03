import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model('next_word_lstm.h5')

with open('tokenizer.pkl', 'rb') as handle:
    tokenizer=pickle.load(handle)

def predict_next_word(model, tokenizer, text, max_len, num_words=10):
    for i in range(num_words):
        token_text = tokenizer.texts_to_sequences([text])
        padded_token_text = pad_sequences(token_text, maxlen=max_len-1, padding='pre')
        prob = np.argmax(model.predict(padded_token_text), axis=-1)
        
        for word, index in tokenizer.word_index.items():
            if index == prob:
                text = text + " " + word
                print(text)
                break
    return text

st.title("Next Word Prediction")
text=st.text_input("Enter the sequence of words")
if st.button("Predict Next Word"):
    max_len = model.input_shape[1] + 1
    next_word =predict_next_word(model, tokenizer, text, max_len, num_words=10)
    st.write(f"Next word:{next_word}")