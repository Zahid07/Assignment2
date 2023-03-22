#create a simple flask app to run php
from flask import Flask, render_template, jsonify, request
import requests
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
app = Flask(__name__ , template_folder='template')

data=[]

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/data')
def get_data():
    # # Dummy data for demonstration purposes
    # data = [
    #     {"title": "Product A", "price": "$10.00", "dateTime": "2022-03-22 10:00:00"},
    #     {"title": "Product B", "price": "$15.00", "dateTime": "2022-03-22 11:00:00"},
    #     {"title": "Product C", "price": "$20.00", "dateTime": "2022-03-22 12:00:00"}
    # ]
    # return jsonify(data) # Return the data as a JSON response
    close_values = []
    
    symbol = 'BTCUSDT'
    interval = '1m'
    limit = 3
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'
    response = requests.get(url)
    for candle in response.json():
        close_values = candle[4]
        open_price = candle[1]
        high_price = candle[2]
        low_price = candle[3]
        volumeto = candle[5]
        volumefrom = candle[7]
        data.append({ "title":"BTCUSDT ", "close_values": close_values, "open_price": open_price, "high_price": high_price, "low_price": low_price, "volumeto": volumeto, "volumefrom": volumefrom})

    #convert data to dataframe
    df = pd.DataFrame(data)
    # get x
    x = df.iloc[:, 1:2].values
    #get y
    y = df.iloc[:, 2:3].values
    #split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    #train model which takes continous values
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    pickle.dump(model, open('btcPredict.pkl', 'wb'))


    
    return jsonify(data) # Return the data as a JSON response

@app.route("/predict", methods=["GET"])
def predict_price():
    # Extract the price input from the form data
    print("Here")
    
    price_input = request.args.get("price")
    #load model
    model = pickle.load(open('btcPredict.pkl', 'rb'))
    #predict
    predicted_price = model.predict([[price_input]])
    return jsonify({"predicted_price": predicted_price[0][0]})
    



if __name__ == '__main__':
    app.run()

