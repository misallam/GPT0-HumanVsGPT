import pickle
import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

def gpt0_search(model_input):
    with open("pkl_assets/gpt0.pkl", "rb") as f:
        gpt0_model = pickle.load(f)

    def encode(model_input):
        embedding = model.encode([model_input])
        doubled_embedding = np.concatenate((embedding[0], embedding[0]))
        return doubled_embedding

    def classify_text(model_input):
        encoded_text = encode(model_input)
        prediction = gpt0_model.predict([encoded_text])

        return "AI-Generated" if prediction[0] == 1 else "Human-Generated"

    return classify_text(model_input)

def main():
    st.title("GPT0 Search Engine")
    query = st.text_input("Enter your query")

    if st.button("Search"):
        results = gpt0_search(query)
        st.write(results)

if __name__ == "__main__":
    main()