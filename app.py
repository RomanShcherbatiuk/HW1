from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import pickle

app = FastAPI()

with open(r'/Users/r.r.shcherbatyuk/Downloads/model_and_scaler.pickle', "rb") as file:
    content = pickle.load(file)

model = content['trained_models']['linear_regression']
scaler = content['scalers']['scaler']

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

class Items(BaseModel):
    objects: List[Item]

def preprocess_data(items: List[Item]) -> pd.DataFrame:
    data = pd.DataFrame([item.dict() for item in items])
    selected_columns = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
    data = data[selected_columns]
    return scaler.transform(data)

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    prediction = model.predict(preprocess_data([item]))[0]
    return prediction

@app.post("/predict_items")
def predict_items(items: Items) -> List[float]:
    data = preprocess_data(items.objects)
    predictions = model.predict(data).tolist()
    return predictions