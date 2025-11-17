import pickle
from fastapi import FastAPI
import uvicorn 
from typing import Dict,Any
from pydantic import BaseModel, Field
from typing import Literal


class Shipment(BaseModel):
    warehouse_block: Literal["A", "B", "C", "D", "F"]
    mode_of_shipment: Literal["Flight", "Ship", "Road"]
    customer_care_calls: int = Field(..., ge=0)
    customer_rating: Literal["low", "very_high", "medium", "very_low", "high"]
    cost_of_the_product: int = Field(..., ge=0)
    prior_purchases: int = Field(..., ge=0)
    product_importance: Literal["low", "medium", "high"]
    discount_offered: int = Field(..., ge=0)
    weight: int = Field(..., ge=1)

    
class PredictResponse(BaseModel):
    delivery_on_time_probability: float
    delivery: bool
    


app = FastAPI(title='delivery-prediction')


with open('delivery-model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)
    



def predict_order(order):
    x = dv.transform([order])
    y_pred = model.predict_proba(x)[0, 1]
    return float(y_pred)
    
    

@app.post('/predict')
def predict(order: Shipment) -> PredictResponse:
    delivery = predict_order(order.dict())
    
    return PredictResponse(
        delivery_on_time_probability = delivery,
        delivery = bool(delivery>=0) 
    )

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=9696)

