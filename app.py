import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

@st.cache_resource
def load_lstm():
    model = load_model('next_word_lstm.h5')
    with open('tokenizer.pickle','rb') as handle:
        tok = pickle.load(handle)
    rev = {v:k for k,v in tok.word_index.items()}
    return model, tok, rev

model, tokenizer, rev_index = load_lstm()

def predict_next_word(model, tokenizer, text, max_sequence_len):
    tok = tokenizer.texts_to_sequences([text])[0]
    if len(tok) >= max_sequence_len:
        tok = tok[-(max_sequence_len-1):]
    tok = pad_sequences([tok], maxlen=max_sequence_len-1, padding='pre')
    pred = model.predict(tok, verbose=0)
    idx = int(np.argmax(pred, axis=1))
    return rev_index.get(idx)

st.title("Next Word Prediction With LSTM And Early Stopping")

input_text = st.text_input("Enter words", "To be or not to")
if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1] + 1
    word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f"Next word: {word}")
