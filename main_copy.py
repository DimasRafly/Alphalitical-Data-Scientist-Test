import os
import re
from fastapi import FastAPI, HTTPException, Request, Form
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from fastapi.templating import Jinja2Templates

# Load saved model & tokenizer
model_save_path = "/Users/dimasrafly/Documents/Test Application/Alphalitical Data Scientist Test/saved_model"  # Path to your model
tokenizer = BertTokenizer.from_pretrained(model_save_path)
model = BertForSequenceClassification.from_pretrained(model_save_path)

# Set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()

# Define the FastAPI app
app = FastAPI()

# Define Jinja2 template folder
templates = Jinja2Templates(directory="templates")

# Clean text function
def clean_text(text):
    text = text.lower()  
    text = re.sub(r'\d+', '', text)  
    text = re.sub(r'[^\w\s]', '', text)  
    text = re.sub(r'\s+', ' ', text)  
    return text

# Predict sentiment function
def predict_sentiment(model, tokenizer, text):
    text = clean_text(text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).cpu().item()

    reverse_label_map = {0: 'positive', 1: 'neutral', 2: 'negative'}
    return reverse_label_map[predicted_class]

# Main page (form)
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Prediction Endpoint
@app.post("/predict/")
async def predict(request: Request, text: str = Form(...)):
    try:
        sentiment = predict_sentiment(model, tokenizer, text)
        return templates.TemplateResponse("index.html", {"request": request, "sentiment": sentiment, "text": text})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))