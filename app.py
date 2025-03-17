from flask import Flask, render_template, request, url_for
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained models and preprocessor
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)
with open("encoder_label.pkl", "rb") as f:
    encoder = pickle.load(f)
with open("dtr1.pkl", "rb") as model_file:
    dtr = pickle.load(model_file)
with open("preprocessor1.pkl", "rb") as f:
    pp1 = pickle.load(f)

# Home page
@app.route("/")
def home():
    return render_template("index.html")




@app.route("/crop_recom", methods=["GET", "POST"])
def crop_recommendation():
# Mapping crops to image filenames    
    crop_images = {
    "apple": "apple.jpg",
    "banana": "banana.jpg",
    "blackgram": "blackgram.jpg",
    "chickpea": "chickpea.jpg",
    "coconut": "coconut.jpg",
    "coffee": "coffee.jpg",
    "cotton": "cotton.jpg",
    "grapes": "grapes.jpg",
    "jute": "jute.jpg",
    "kidneybeans": "kidneybeans.jpg",
    "lentil": "lentil.jpg",
    "maize": "maize.jpg",
    "mango": "mango.jpg",
    "mothbeans": "mothbeans.jpg",
    "mungbean": "mungbean.jpg",
    "muskmelon": "muskmelon.jpg",
    "orange": "orange.jpg",
    "papaya": "papaya.jpg",
    "pigeonpeas": "pigeonpeas.jpg",
    "pomegranate": "pomegranate.jpg",
    "rice": "rice.jpg",
    "watermelon": "watermelon.jpg"}
    

    
    

    if request.method == "POST":
        N = float(request.form["N"])
        P = float(request.form["P"])
        K = float(request.form["K"])
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])
        
        input1 = [N, P, K, temperature, humidity, ph, rainfall]
        prediction = model.predict(np.array(input1).reshape(1, -1))
        decoded_prediction = encoder.inverse_transform([prediction[0]])

        crop_name = decoded_prediction[0].lower()
        crop_image = crop_images.get(crop_name, "default.jpg")  # Default image if crop not found
        
        return render_template("crop_recommendation.html", prediction=decoded_prediction, crop_image=crop_image)

    return render_template("crop_recommendation.html", prediction=None, crop_image=None)



# Crop Yield Prediction Route
@app.route("/crop_yield", methods=["GET", "POST"])
def crop_yield_prediction():
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
            transformed_features = pp1.transform(features)
            predicted_yield = dtr.predict(transformed_features).reshape(1, -1)
            return predicted_yield[0] * 100

        result = prediction(Crop, Crop_Year, Season, State, Area, Production, Annual_Rainfall, Fertilizer, Pesticide)
        result = f"{float(result):.2f}"  # Format to 2 decimal places
        
        return render_template("crop_yield.html", prediction=result)
    
    return render_template("crop_yield.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
