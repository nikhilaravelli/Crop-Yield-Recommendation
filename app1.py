from flask import Flask, render_template, request
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the trained model and preprocessor
with open("dtr.pkl", "rb") as model_file:
    dtr = pickle.load(model_file)
with open("preprocessor.pkl","rb") as f:
    pp1=pickle.load(f)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        Crop = request.form["Crop"]
        Crop_Year = int(request.form["Crop_Year"])
        Season = request.form["Season"]
        State = request.form["State"]
        Area = float(request.form["Area"])
        Production = float(request.form["Production"])
        Annual_Rainfall = float(request.form["Annual_Rainfall"])
        Fertilizer = float(request.form["Fertilizer"])
        Pesticide = float(request.form["Pesticide"])
        def prediction(Crop, Crop_Year, Season, State, Area, Production, Annual_Rainfall, Fertilizer, Pesticide):
            features = np.array([[Crop, Crop_Year, Season, State, Area, Production, Annual_Rainfall, Fertilizer, Pesticide]], dtype=object)
            features = np.array(features)  # Ensure the correct type
            transformed_features = pp1.transform(features)
            predicted_yield = dtr.predict(transformed_features).reshape(1, -1)
            return predicted_yield[0]*100
        

        result=prediction(Crop, Crop_Year, Season, State, Area, Production, Annual_Rainfall, Fertilizer, Pesticide)
        result = f"{float(result[0]):.2f}"  
        return render_template("index.html", prediction=result)
    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
