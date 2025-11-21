from flask import Flask, request, render_template
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import pandas as pd

# ------------------------------
# Load tokenizer
# ------------------------------
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# ------------------------------
# Define multimodal model (title + review)
# ------------------------------
class MultiModalSentimentModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.title_bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.review_bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768*2, 2)

    def forward(self, input_ids_title, attention_mask_title, input_ids_review, attention_mask_review):
        title_output = self.title_bert(input_ids=input_ids_title, attention_mask=attention_mask_title)
        review_output = self.review_bert(input_ids=input_ids_review, attention_mask=attention_mask_review)

        title_cls = title_output.last_hidden_state[:,0,:]
        review_cls = review_output.last_hidden_state[:,0,:]

        combined = torch.cat((title_cls, review_cls), dim=1)
        x = self.dropout(combined)
        logits = self.classifier(x)
        return logits

# ------------------------------
# Load trained weights
# ------------------------------
model = MultiModalSentimentModel().to(device)
model.load_state_dict(torch.load("multimodal_sentiment_model.pt", map_location=device))
model.eval()

# ------------------------------
# Prediction function
# ------------------------------
def predict_sentiment(title, review):
    title_enc = tokenizer(title, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
    review_enc = tokenizer(review, truncation=True, padding='max_length', max_length=128, return_tensors='pt')

    input_ids_title = title_enc['input_ids'].to(device)
    attention_mask_title = title_enc['attention_mask'].to(device)
    input_ids_review = review_enc['input_ids'].to(device)
    attention_mask_review = review_enc['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(input_ids_title, attention_mask_title, input_ids_review, attention_mask_review)
        pred = torch.argmax(logits, dim=1).item()
    return "Positive" if pred == 1 else "Negative"  # 0 = Negative, 1 = Positive

# ------------------------------
# Flask app
# ------------------------------
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    table_data = None
    if request.method == "POST":
        if 'review_text' in request.form and 'title_text' in request.form:
            title = request.form['title_text']
            review = request.form['review_text']
            result = predict_sentiment(title, review)
        elif 'csv_file' in request.files:
            file = request.files['csv_file']
            df = pd.read_csv(file, header=None, names=['label', 'title', 'review'])
            df['Predicted Sentiment'] = df.apply(lambda row: predict_sentiment(row['title'], row['review']), axis=1)
            table_data = df.to_dict(orient='records')
    return render_template("index.html", result=result, table_data=table_data)

if __name__ == "__main__":
    app.run(debug=True)
