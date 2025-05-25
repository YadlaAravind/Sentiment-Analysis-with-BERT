# 🧠 Sentiment Analysis using BERT

A deep learning project that uses the BERT transformer model to classify the sentiment of text into **positive**, **neutral**, or **negative**. Built using Hugging Face Transformers and PyTorch, this project demonstrates how to fine-tune a pre-trained BERT model for sentiment classification.

---

## 📌 Features

- ✅ 3-class sentiment classification (positive, neutral, negative)
- 🧠 Model: `bert-base-uncased` from Hugging Face
- 🔁 Fine-tuned on custom synthetic dataset
- 📊 Evaluation using accuracy and classification report
- 🧪 Custom sentence inference included

---

## 🗂️ Dataset

The dataset is synthetically generated and includes over 50 text samples with sentiment labels. The data is saved in a CSV file named `sample_data.csv` with the following format:

```csv
text,sentiment
"This product is amazing! I love it.",positive
"Not worth the money.",negative
"Neutral review without strong opinions.",neutral
...
````

---

## 🚀 How It Works

1. Load and preprocess data
2. Tokenize using BERT tokenizer
3. Convert to input tensors with attention masks
4. Train using `AdamW` optimizer and learning rate scheduler
5. Evaluate using scikit-learn metrics
6. Predict sentiment for a custom test sentence

---

## ▶️ Usage

### 1. Install Requirements

```bash
pip install pandas numpy torch scikit-learn transformers tqdm
```

### 2. Run the Script

```bash
python sentimental.py
```

Expected output:

```
Saved 51 samples to sample_data.csv
Accuracy: 0.82
Classification Report:
              precision    recall  f1-score   support
    negative       0.80      0.83      0.81
     neutral       0.75      0.75      0.75
    positive       0.90      0.89      0.89
Predicted Sentiment: positive
```

---

## ⚙️ Model Inference

The script includes code to test sentiment on a new sentence:

```python
test_sentence = "Completely satisfied with my purchase"
```

The model outputs:

```
Predicted Sentiment: positive
```

---

## 🧰 Technologies Used

* Python 3.x
* PyTorch
* Hugging Face Transformers
* scikit-learn
* pandas, NumPy
* tqdm

---

## 💻 Hardware Support

The script automatically uses GPU (`cuda`) if available for faster training and inference. Falls back to CPU if not.

---

## 📈 Future Improvements

* Integrate with a Streamlit/Flask web UI
* Use real-world datasets (IMDB, Yelp, etc.)
* Save and load fine-tuned model for reuse
* Deploy as an API

---

## 👨‍💻 Author

Developed by Yadla Aravind (https://github.com/YadlaAravind)
Feel free to fork, contribute, or open issues!
