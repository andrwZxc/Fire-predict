from flask import Flask, request, render_template, jsonify, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from PIL import Image
import numpy as np
import os
import pickle 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from flask import send_from_directory



app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False


# # Загрузка модели
# model_path = 'models/model.pkl'
# if os.path.exists(model_path):
#     with open(model_path, 'rb') as f:
#         model = pickle.load(f)
# else:
#     raise FileNotFoundError(f"Model file not found at {model_path}")

with open('models/model15.05.pkl', 'rb') as f:
    artifacts = pickle.load(f)
    print("Загруженные фичи: ", artifacts['features'])
    model = artifacts['model']
    #le = artifacts['label_encoder']
    features = artifacts['features']
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

@app.route('/get_csv_data')
def get_csv_data():
    return send_from_directory('static/datasets', 'TEST15.05.csv')
    

@app.route('/predict_fire', methods=['GET','POST'])
def predict_fire():
    if request.method == 'GET':
        return render_template('prediction.html')  # если метод get то просто загружаем страницу
    
    try:
        print("Получен запрос с данными:", request.json)
        # Получаем данные из запроса
        data = request.get_json() # преобразование данных из JSON в py словарь
        
        

    
        # Преобразуем в DataFrame (в том же порядке, как при обучении)
        input_data = pd.DataFrame([[
            data['CO2_Room'],
            data['H2_Room'],
            data['PM05_Room'],
            data['PM100_Room'],
            data['PM10_Room'],
            data['PM25_Room'],
            data['PM40_Room'],
            data['PM_Room_Typical_Size'],
            data['PM_Total_Room'],
            data['VOC_Room_RAW'],
            data['Temperature_Room'],
            data['Humidity_Room'],
            data['CO_Room'],
            
        ]], columns=['CO2_Room','H2_Room', 'PM05_Room' ,'PM100_Room', 'PM10_Room', 'PM25_Room' , 'PM40_Room', 'PM_Room_Typical_Size' ,'PM_Total_Room', 'VOC_Room_RAW' ,'Temperature_Room', 'Humidity_Room','CO_Room' ])
        
        print("Данные для предсказания: ", input_data)
    
        #input_data['Sensor_ID'] = le.transform(input_data['Sensor_ID']) # перекодируем метки

        input_data = input_data[features]


        # Делаем предсказание
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]  # Вероятность класса 1 (пожар)
        
        print("Результат предсказания: ", prediction, " === ", probability)

        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability),
            'status': 'success'
        })
        
    except Exception as e:
        print("Ошибка: ", str(e))
        return jsonify({'error': str(e), 'status':'error'})
    
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
    app.run(debug=True, port=5000)
