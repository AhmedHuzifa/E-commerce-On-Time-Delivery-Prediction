Project Overview: 
This project develops a machine learning model to predict on-time delivery for e-commerce shipments. The goal is to help the planning department identify orders that are at higher risk of delay and take corrective action.
The dataset contains 10,999 historical orders, with features such as warehouse block, shipment mode, customer rating, prior purchases, discount offered, and product weight.


Primary users: 
The Planning department shall use the web service to make data-driven decisions regarding deliveries, whether they would be delivered on time or not, and what could be done to improve the order delivery reliability.


Exploratory Data Analysis:

-Features
Warehouse block: Five sections in the warehouse (A, B, C, D, F).
Mode of shipment: Three modes — Ship, Flight, Road.
Customer care calls: Number of customer service calls per order.
Customer rating: Satisfaction rating from 1 (Worst) to 5 (Best). The integer data type is later changed to strings (very_high, high, medium, low, very_low) as they represent categories. 
Cost of the product: Price in USD.
Prior purchases: Number of previous purchases by the same customer.
Product importance: Product classification (Low, Medium, High).
Discount offered: Discount applied to the order.
Weight: Product weight in grams.
Target variable: On-time delivery → Binary (1 = On time, 0 = Delayed)

-Delivery rate
Delivered on time: 6,563 orders
Proportion: ~59.7%

-Correlation with target
Discount offered	+0.397
Weight	–0.264
Customer care calls	–0.072
Cost of product	–0.078
Prior purchases	–0.060

-Mutual Feature Information
warehouse_block       0.000182
mode_of_shipment      0.000045
customer_rating       0.000283
product_importance    0.000649

-insights
Discounted items are more likely to be delivered on time.
Heavier products are significantly more likely to be delayed.
Categorical features (warehouse block, product importance, shipment mode) show very small information gains.

Feature Engineering:
Encoding: One-Hot Encoding for all categorical features.
Scaling: Not required for tree-based models and classification.
Train/Validation/test split: Stratified 60% / 20% / 20%

Models Trained (Baseline Comparison):
Model	                AUC
Logistic Regression  	0.6920
Decision Tree   	    0.7139
XGBoost	                0.7217
Random Forest	        0.7283

Best baseline model: 
Random Forest Classifier 
with Hyperparameters {"max_depth": 10, "min_samples_leaf": 10, "n_estimators": 100}

Final performance:
AUC: 0.7458 (on full 80% training set)

Serving:
FastAPI is used as the web framework.

Environment Management:
UV is used to create a reproducible virtual environment.
uv.lock ensures deterministic dependency installs.
to run the app in the environment: 
uv run uvicorn predict:app --host 0.0.0.0 --port 9696 --reload

Containerization:
Complete application packaged in a Docker image.
Guarantees reproducibility and easy deployment.
to build a docker image:
docker build -t delivery-prediction .
then run the  the image:
docker run -it --rm -p 9696:9696 delivery-prediction
