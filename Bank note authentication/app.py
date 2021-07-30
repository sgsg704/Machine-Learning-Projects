import uvicorn
from fastapi import FastAPI
from BankNotes import BankNote
import numpy as np
import pandas as pd
import pickle


app=FastAPI()
pickle_in=open("classifier.pkl",'rb')
classifier=pickle.load(pickle_in)


@app.get('/')
def index():
    return {'message':'Hello world'}

@app.get('/{name}')
def get_name(name:str):
    return {"Welcome to my world ": f'{name}'}


@app.post('/predict')
def predict_banknote(data:BankNote):
    data=data.dict()
    variance=data["variance"]

    skewness=data["skewness"]
    
    kurtosis=data["kurtosis"]
    
    entropy=data["entropy"]
    prediction=classifier.predict([[variance,skewness,kurtosis,entropy]])

    if prediction[0]>0.5:
        prediction="Fake Note"
    else:
        prediction="Real Note"

    return {"prediction":prediction}
if __name__=='__main__':
    uvicorn.run(app,host='127.0.0.1',port=8000)



 