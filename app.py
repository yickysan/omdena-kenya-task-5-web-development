from flask import Flask, render_template, url_for, request
from OmdenaKenyaRoadAccidents.pipeline.prediction_pipeline import PredictPipeline

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analysis")
def analysis():
    return render_template("analysis.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    return render_template("predict.html")

@app.route("/predictions", methods=["GET", "POST"])
def make_prediction():
    if request.method == "GET":
        return render_template("predict.html")
    
    else:
                
        pipeline = PredictPipeline(age_band_of_driver = request.form.get("driver-age"),
                                   driving_experience = request.form.get("driving-experience"),
                                   defect_of_vehicle = request.form.get("vehicle-defect") ,
                                   light_conditions = request.form.get("light-condition"),
                                   number_of_vehicles_involved = request.form.get("no-of-vehicles-involved"),
                                   cause_of_accident = request.form.get("cause-of-accident")
                                   )
                

        result = pipeline.predict()[0]
        result_p = f"The predicted severity of the accident is a {result}"

    
    return render_template("predict.html", result_p=result_p)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")