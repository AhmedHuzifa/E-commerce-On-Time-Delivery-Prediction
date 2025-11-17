
import requests


url = 'http://localhost:9696/predict'

order = {
  "warehouse_block": "F",
  "mode_of_shipment": "Flight",
  "customer_care_calls": 3,
  "customer_rating": "very_low",
  "cost_of_the_product": 162,
  "prior_purchases": 0,
  "product_importance": "medium",
  "discount_offered": 0,
  "weight": 1417
}


resopnse = requests.post(url, json=order)
delivery = resopnse.json()


print('response:', delivery)



if delivery['delivery_on_time_probability'] >= 0.5:
    print('This Order will be delivered on time')
else:
    print('This Order will not be delivered on time')
