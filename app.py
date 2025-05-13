from flask import Flask, request, render_template, jsonify, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from PIL import Image
import numpy as np
import os
import pickle 
import tensorflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

#filename = 'model.pkl'
#with open(filename, 'rb') as file: # конструкция для безопасности, чтобы файл закрылся после выполнения блока кода
#    model = pickle.load(file) # read binary


app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False


# Загрузим модель
model = load_model('models/model.h5')
with open('models/preprocessors.pkl','rb') as f:
    preprocessors = pickle.load(f)
    scaler = preprocessors['scaler']
    label_encoders = preprocessors['label_encoders']



db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)


def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.route('/')
def landing():
    return render_template('landing.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']  # Store password as plain text

        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('signup'))

        new_user = User(username=username, password=password)  # No hashing here
        db.session.add(new_user)
        db.session.commit()
        flash('Signup successful. Please login.')
        return redirect(url_for('login'))
    
    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()
        if user and user.password == password:  
            session['user_id'] = user.id
            flash('Login successful')
            return redirect(url_for('predict_fire'))
        else:
            flash('Invalid credentials')
    
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('Logged out successfully')
    return redirect(url_for('landing'))


@app.route('/predict_fire', methods=['GET','POST'])
def predict_fire():
    if request.method == 'GET':
        return render_template('prediction.html')  # если метод get то просто загружаем страницу
    elif request.method == 'POST':return redirect(url_for('login.html'))
    try:
        # Получаем данные из запроса
        data = request.json # преобразование данных из JSON в py словарь
        
        # Преобразуем в DataFrame (в том же порядке, как при обучении)
        input_data = pd.DataFrame([[
            data['temperature'],
            data['humidity'],
            data['tvoc'],
            data['eco2'],
            data['raw_h2'],
            data['raw_ethanol'],
            data['pressure']
        ]], columns=['Temperature[C]', 'Humidity[%]', 'TVOC[ppb]', 
                    'eCO2[ppm]', 'Raw H2', 'Raw Ethanol', 'Pressure[hPa]'])
        
        # Масштабируем данные (если нужно)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        scaled_data = scaler.transform(input_data)
        
        # Делаем предсказание
        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0][1]  # Вероятность класса 1 (пожар)
        
        return jsonify({
            'prediction': int(prediction),
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400
    
    # features = np.array(data['features']).reshape(1, -1)
        
    #     # Препроцессинг
    #     scaled_features = scaler.transform(features)
        
    #     # Предсказание
    #     prediction = model.predict(scaled_features)
        
    #     # Постобработка (если нужно декодировать метки)
    #     result = {
    #         'fire_probability': float(prediction[0][0]),
    #         'fire_class': int(prediction[0][0] > 0.5)
    #     }
        
    #     return jsonify(result)
    
    # except Exception as e:
    #     return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
