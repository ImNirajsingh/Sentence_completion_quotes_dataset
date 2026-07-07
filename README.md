# ✍️ Sentence Completion & Quote Generation using LSTMs

An interactive Deep Learning project built in a Jupyter Notebook that trains an AI model to generate text and complete sentences based on a dataset of quotes. 

This project demonstrates a complete Natural Language Processing (NLP) pipeline—from text preprocessing and tokenization to training a Long Short-Term Memory (LSTM) neural network and saving the model artifacts for future deployment.

---

## 📌 Project Overview

The objective of this notebook is to process human language data and build a predictive text model. Given a seed phrase (e.g., *"life is about"*), the trained LSTM network predicts the most logically sound sequence of following words, effectively generating original quotes.

---
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/6879af77-e58c-46de-9829-7c176277905b" />


## 🛠️ Technology Stack & Libraries

This notebook utilizes a **Python 3** environment (optimized for GPU acceleration, e.g., on Google Colab) and relies on the following libraries:

* **Data Handling:** `pandas`, `numpy`
* **Text Processing:** `string` (punctuation removal)
* **Deep Learning (TensorFlow / Keras):**
  * `Tokenizer`: For converting text into integer sequences.
  * `Sequential`, `LSTM`, `Dense`, `Embedding`: For building the neural network architecture.
* **Artifact Serialization:** `pickle` (for saving the tokenization rules and sequence lengths).

---

## 📊 Dataset Overview

The project relies on `qoute_dataset.csv`, which contains a collection of text quotes. 

**Preprocessing Steps Performed:**
1. **Extraction:** Isolating the text data from the DataFrame.
2. **Standardization:** Converting all strings to lowercase.
3. **Cleaning:** Stripping out all punctuation using Python's `str.maketrans` to ensure the model focuses strictly on word vocabulary rather than syntax marks.

---

## 🚀 Notebook Workflow

### 1. Data Cleaning & Tokenization
* Cleans the text corpus and feeds it into a Keras `Tokenizer` (capped at a `vocab_size` of 10,000 words).
* Maps every unique word to a specific integer index to create a numerical dictionary.

### 2. Sequence Generation
* Generates input sequences and pads them to a uniform length (`max_len`).
* Prepares predictors (X) and labels (Y) for the text prediction task.

### 3. LSTM Model Training
* Constructs and trains a Recurrent Neural Network (RNN) using LSTM layers, which are uniquely suited for understanding context and memory in sequential text data.

### 4. Text Generation 
* Features a custom `generate_text` function. 
* By inputting a `seed_text` (like *"life is about"*) and specifying the number of words to generate, the model iteratively predicts the next word in the sentence.

### 5. Exporting Assets
* Saves the fitted tokenizer (`tokenizer.pkl`) and sequence length integer (`max_len.pkl`) locally using `pickle`. This ensures that any new text inputs during future inferences are processed exactly the same way as the training data.

---

## 💻 How to Run the Notebook Locally

1. **Clone or Download the Repository:** Ensure `qoute_dataset.csv` and the `.ipynb` notebook file are in the same directory.
2. **Install Dependencies:**
   ```bash
   pip install numpy pandas tensorflow matplotlib seaborn
