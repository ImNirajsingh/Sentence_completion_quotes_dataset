import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------------------------------
# Page Configuration
# -----------------------------------------------------
st.set_page_config(
    page_title="Next Sentence Predictor",
    page_icon="🧠",
    layout="centered"
)

# -----------------------------------------------------
# Custom CSS
# -----------------------------------------------------
st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

h1{
    text-align:center;
    color:#1E3A8A;
}

.description{
    text-align:center;
    color:#555;
    font-size:18px;
    margin-bottom:30px;
}

.stButton>button{
    width:100%;
    height:55px;
    border-radius:10px;
    font-size:18px;
    font-weight:bold;
    background-color:#2563EB;
    color:white;
}

.stButton>button:hover{
    background-color:#1D4ED8;
    color:white;
}

.generated-box{
    background:white;
    border-radius:12px;
    padding:20px;
    border-left:6px solid #2563EB;
    box-shadow:0px 3px 10px rgba(0,0,0,.15);
    font-size:22px;
    color:black;
    line-height:1.8;
    font-weight:500;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:30px;
    font-size:14px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# Load Model
# -----------------------------------------------------
@st.cache_resource
def load_assets():

    model = load_model("lstm_model.h5")

    with open("tokenizer.pkl","rb") as f:
        tokenizer = pickle.load(f)

    with open("max_len.pkl","rb") as f:
        max_len = pickle.load(f)

    return model, tokenizer, max_len


model, tokenizer, max_len = load_assets()

# -----------------------------------------------------
# Predictor
# -----------------------------------------------------
def predictor(model, tokenizer, text, max_len, temperature=0.8):

    sequence = tokenizer.texts_to_sequences([text])[0]

    sequence = pad_sequences(
        [sequence],
        maxlen=max_len,
        padding="pre"
    )

    prediction = model.predict(sequence, verbose=0)[0]

    if np.max(prediction) < 0.20:
        return ""

    prediction = np.log(prediction + 1e-8) / temperature
    prediction = np.exp(prediction)
    prediction = prediction / np.sum(prediction)

    predicted_index = np.random.choice(
        len(prediction),
        p=prediction
    )

    for word, index in tokenizer.word_index.items():

        if index == predicted_index:
            return word

    return ""


# -----------------------------------------------------
# Generate Text
# -----------------------------------------------------
def generate_text(model, tokenizer, seed_text, max_len, num_words):

    generated = []

    for _ in range(num_words):

        next_word = predictor(
            model,
            tokenizer,
            seed_text,
            max_len
        )

        if next_word == "":
            break

        if len(generated) >= 2:
            if generated[-1] == generated[-2] == next_word:
                break

        generated.append(next_word)

        seed_text += " " + next_word

        if next_word in [".", "!", "?"]:
            break

    seed_text = seed_text.strip()

    seed_text = seed_text.capitalize()

    if not seed_text.endswith((".", "!", "?")):
        seed_text += "."

    return seed_text

# -----------------------------------------------------
# UI
# -----------------------------------------------------
st.markdown("<h1>🧠 Next Sentence Predictor</h1>", unsafe_allow_html=True)

st.markdown(
"""
<div class='description'>
Generate meaningful sentence completions using a trained LSTM Deep Learning model.
</div>
""",
unsafe_allow_html=True
)

seed_text = st.text_input(
    "Enter a starting sentence",
    placeholder="Example: Life is about"
)

num_words = st.slider(
    "Number of words to generate",
    min_value=5,
    max_value=50,
    value=20
)

if st.button("🚀 Generate Sentence"):

    if seed_text.strip() == "":
        st.warning("Please enter a sentence.")
    else:

        with st.spinner("Generating..."):

            result = generate_text(
                model,
                tokenizer,
                seed_text,
                max_len,
                num_words
            )

        st.markdown("## ✨ Generated Sentence")

        st.markdown(
            f"""
            <div class="generated-box">
            {result}
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown(
"""
<div class='footer'>
Built with ❤️ Niraj Singh, IIT Guwahati <a href='
</div>
""",
unsafe_allow_html=True
)
