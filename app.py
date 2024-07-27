import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load the saved model and encoders
model = joblib.load('travel_mode_model.pkl')
encoder = joblib.load('onehot_encoder.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.get("/", response_class=HTMLResponse)
def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
def predict_travel_mode(request: Request,
                        terrain_type: str = Form(...),
                        distance_km: float = Form(...),
                        terrain_difficulty: str = Form(...),
                        weather_conditions: str = Form(...),
                        accessibility: str = Form(...),
                        elevation: str = Form(...),
                        travel_time_hrs: float = Form(...),
                        cost: float = Form(...)):
    # Convert input data to DataFrame
    input_data = {
        'Terrain Type': terrain_type,
        'Distance (km)': distance_km,
        'Terrain Difficulty': terrain_difficulty,
        'Weather Conditions': weather_conditions,
        'Accessibility': accessibility,
        'Elevation': elevation,
        'Travel Time (hrs)': travel_time_hrs,
        'Cost ($)': cost
    }
    input_df = pd.DataFrame([input_data])

    # Apply one-hot encoding to the new data
    encoded_new_data = encoder.transform(input_df[['Terrain Type', 'Terrain Difficulty', 'Weather Conditions', 'Accessibility', 'Elevation']])
    encoded_new_df = pd.DataFrame(encoded_new_data, columns=encoder.get_feature_names_out())

    # Combine the encoded new data with the rest of the new data
    input_df = input_df.drop(['Terrain Type', 'Terrain Difficulty', 'Weather Conditions', 'Accessibility', 'Elevation'], axis=1)
    input_df = pd.concat([input_df, encoded_new_df], axis=1)

    # Predict the travel mode
    predicted_travel_mode = model.predict(input_df)
    
    # Convert the numerical prediction back to the original label
    predicted_travel_mode_label = label_encoder.inverse_transform(predicted_travel_mode)

    return templates.TemplateResponse("result.html", {"request": request, "predicted_travel_mode": predicted_travel_mode_label[0]})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
