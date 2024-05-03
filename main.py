import numpy as np
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle

# Create FastAPI instance
app = FastAPI()
templates = Jinja2Templates(directory="templates")

import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Load the pickled SVM classifier
with open("svm_classifier.pkl", "rb") as f:
    model = pickle.load(f)

# Define the route for the home page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Define the route for prediction
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, Sepal_Length: float = Form(...), Sepal_Width: float = Form(...),
                  Petal_Length: float = Form(...), Petal_Width: float = Form(...)):
    try:
        features = np.array([[Sepal_Length, Sepal_Width, Petal_Length, Petal_Width]])
        prediction = model.predict(features)
        return templates.TemplateResponse("index.html", {"request": request, "prediction_text": f"The flower species is {prediction}"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
