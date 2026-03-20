import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
application=Flask(__name__)
app=application
# import ridge model and standard scaler
ridge_model=pickle.load(open('./model/ridge.pkl','rb'))
standard_scaler=pickle.load(open('./model/scaler.pkl','rb'))
@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predictdata",methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        print("POST")
        Temperature=float(request.form.get('Temperature'))
        WS=float(request.form.get('WS'))
        ISI=float(request.form.get('ISI'))
        Rain=float(request.form.get('Rain'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        RH=float(request.form.get('RH'))
        new_data_scaled=standard_scaler.transform([[Temperature,RH,WS,Rain,FFMC,DMC,ISI,Classes,Region]])
        print("new_data_scaled ",new_data_scaled)
        result=ridge_model.predict(new_data_scaled)
        print("result ",result)
        return render_template('home.html',results=result[0])
    else:
        return render_template('home.html')
if __name__=="__main__":
    app.run(host='0.0.0.0')