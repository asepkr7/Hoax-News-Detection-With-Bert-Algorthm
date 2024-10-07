from flask import Flask, render_template, request, jsonify, redirect, url_for
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)
# python -m flask run
# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
model = AutoModelForSequenceClassification.from_pretrained("model")
model.eval()

def predict_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    
    # Mengubah hasil prediksi dari angka menjadi kategori
    if predicted_class == 0:
        prediction_result = "asli"
    else:
        prediction_result = "hoax"
    
    return prediction_result

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    predicted_class = predict_text(text)
    
    # Mengirim kembali hasil prediksi dalam format yang diinginkan
    if predicted_class == "asli":
        result = "asli"
    else:
        result = "hoax"
    
    return jsonify({'predicted_class': result,  'input_text': text})

@app.route('/result', methods=['GET'])
def result():
    text = request.args.get('text')
    prediction = request.args.get('prediction')
    return render_template('result.html', text=text, prediction=prediction)

if __name__ == '__main__':
    create_app().run()
    app.run(debug=True)
