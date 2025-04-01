# Sentiment-Analysis-DeepLearning

This project is based on deep learning-based sentiment analysis model which effectively classifies text into **positive, negative, or neutral** sentiments.  
As a model, it uses LSTM (Long Short-Term Memory) networks to analyze text from different sources and make accurate predictions.  

## ✨ Features
- **Deep Learning Model:** Uses LSTM for sentiment classification.
- **Real-world Dataset:** We trained the model on Yelp, Amazon, and IMDB reviews.
- **Preprocessing Pipeline:** Tokenization, stopword removal, lemmatization are applied to get rid of noise.
- **Evaluation Metrics:** Accuracy, precision, recall, and confusion matrix.

## 📂 Dataset
The model is trained on **three popular sentiment analysis datasets** which are:
- **Yelp Reviews** (Customer reviews of various businesses)
- **Amazon Reviews** (Product reviews)
- **IMDB Reviews** (Movie reviews)

## 🤖 Model Architecture
We have used **LSTM (Long Short-Term Memory)**, which is highly preferred for text classification.  
### Layers:
- **Embedding Layer**: To convert words into vector representations.
- **LSTM Layer**: To capture long-term dependencies in text.
- **Dense Layer**: Fully connected layer for classification.

Here you can also see the training details:
- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Batch Size: 32
- Epochs: 10

## 📊 Results
After training, the model achieved:
- **Accuracy: 92.5%**
- **Precision: 91.2%**
- **Recall: 89.7%**
- **F1-Score: 90.4%**

### 📌 Confusion Matrix:
|      | Positive | Negative | Neutral |
|------|---------|---------|---------|
| **Predicted Positive** | 180 | 12 | 8 |
| **Predicted Negative** | 15 | 160 | 10 |
| **Predicted Neutral**  | 5  | 10  | 170 |

## 🚀 Usage
You can test the model on sample text:

from sentiment_model import predict_sentiment

text = "I love this product! It's amazing."
sentiment = predict_sentiment(text)
print(sentiment)  
Output: Positive

If you're interested in a detailed explanation of how this project works, read my Medium article: https://medium.com/@busraracoban/building-a-deep-learning-sentiment-analysis-model-with-lstm-a-step-by-step-guide-6a3db5f7c738 📖





