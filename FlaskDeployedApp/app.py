import os
from flask import Flask, redirect, render_template, request, jsonify
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd
from pathlib import Path
import requests
import json
import torchvision.transforms as transforms


disease_info = pd.read_csv('disease_info.csv')
supplement_info = pd.read_csv('supplement_info.csv')

model = CNN.CNN(39)    
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
model.eval()

def prediction(image_path):
    image = Image.open(image_path).convert("RGB")
    img = transform(image)
    img = img.unsqueeze(0)
    print(img.shape)
    #image = image.resize((224, 224))
    #input_data = TF.to_tensor(img)
    input_data = img.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index


app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 224x224 boyutuna getir
    transforms.ToTensor(),  # Tensor formatına çevir
])

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        print(request.files)
        print(request.files["image"])
        print(request.files["image"].filename)
        image = request.files['image']
        filename = image.filename
        print(filename)
        file_path = os.path.join('static\\uploads', filename)
        image.save(file_path)
        print(file_path)
        pred = prediction(file_path)
        title = disease_info['disease_name'][pred]
        description =disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        return render_template('submit.html' , title = title , desc = description , prevent = prevent , 
                               image_url = image_url , pred = pred ,sname = supplement_name , simage = supplement_image_url , buy_link = supplement_buy_link)

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image = list(supplement_info['supplement image']),
                           supplement_name = list(supplement_info['supplement name']), disease = list(disease_info['disease_name']), buy = list(supplement_info['buy link']))

def generate_response(prompt, model="tarim-muhendisi"):
    response = requests.post('http://localhost:11434/api/generate', 
                             json={
                                 "model": model,
                                 "prompt": prompt
                             })
    print(response)

    #return response.json().get('response', 'Yanıt alınamadı.')
    response.encoding = "utf-8"
    return response.text

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')

    if not user_message:
        return jsonify({'response': 'Lütfen bir mesaj girin.'}), 400

    # Bitki hastalıkları ile ilgili yanıt üret
    response = generate_response(user_message)
    response = response.rstrip('\n')
    out = ""
    for x in response.split('\n'):
        out += json.loads(x)["response"]

    return jsonify({'response': out})

@app.route('/chat')
def chat_():
    return render_template('chat.html')

if __name__ == '__main__':
    app.run(debug=True)

