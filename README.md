\# MLOps Project



\## Project Overview



This project demonstrates a machine learning workflow for predicting house prices in Indian Rupees using Linear Regression.



The project includes dataset preparation, preprocessing, model training, evaluation, MLflow experiment tracking, Git version control, and FastAPI deployment.



\## Technologies Used



\- Python

\- Pandas and NumPy

\- Scikit-learn

\- MLflow

\- Joblib

\- FastAPI

\- Uvicorn

\- Git

\- SQLite



\## Dataset



The dataset is stored at `data/house\_prices.csv`.



\*\*Input features:\*\*

\- `area\_sqft` — House area in square feet

\- `bedrooms` — Number of bedrooms

\- `bathrooms` — Number of bathrooms

\- `house\_age` — Age of the house in years



\*\*Target variable:\*\*

\- `price` — House price in Indian Rupees



\## Model



Algorithm: Linear Regression



Training/testing split: 80% / 20%



\## Evaluation Results



| Metric | Value |

|---|---:|

| Mean Absolute Error | ₹174,633.18 |

| Root Mean Squared Error | ₹191,509.98 |

| R² Score | 0.9973 |



These results were obtained on the project's test dataset and do not establish accuracy on real-world property transactions.



\## Training



Activate the project environment and run:



```cmd

python train\_model.py

```



The script trains and evaluates the model, records the experiment in MLflow, and saves the trained model using Joblib.



\## MLflow Experiment Tracking



Start MLflow:



```cmd

mlflow ui --backend-store-uri sqlite:///E:/prime-skill\_AI/mlflow.db --port 5001

```



Open `http://127.0.0.1:5001`.



Experiment name: \*\*House Price Prediction\*\*



\## FastAPI Deployment



Start the API:



```cmd

python -m uvicorn app:app --reload --port 8000

```



Open Swagger UI:



`http://127.0.0.1:8000/docs`



\### Prediction Endpoint



`POST /predict`



Example request:



```json

{

&#x20; "area\_sqft": 100,

&#x20; "bedrooms": 2,

&#x20; "bathrooms": 1,

&#x20; "house\_age": 5

}

```



The API returns the predicted house price in INR.



\## Version Control



Git is used to track project source code and changes. Generated model files and local experiment data are excluded from Git by default.



\## Conclusion



The project implements an end-to-end local MLOps workflow, from data preparation through model training, experiment tracking, model persistence, and API-based predictions.

