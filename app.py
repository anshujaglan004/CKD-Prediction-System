
from flask import Flask, request, render_template_string
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load the trained Decision Tree model
model = joblib.load("kidney_model.pkl")

# Get feature columns directly from the trained model
if not hasattr(model, "feature_names_in_"):
    raise ValueError(
        "Feature names are not available in the model. "
        "Please check how the model was trained."
    )

feature_columns = list(model.feature_names_in_)

# Numerical input fields
numeric_fields = [
    "age", "bp", "sg", "al", "su", "bgr", "bu",
    "sc", "sod", "pot", "hemo", "pcv", "wc", "rc"
]

# Categorical input fields
categorical_fields = [
    "rbc", "pc", "pcc", "ba", "htn",
    "dm", "cad", "appet", "pe", "ane"
]

# Get category options from the model's feature names
input_fields = []

for field in numeric_fields:
    if field in feature_columns:
        input_fields.append({
            "name": field,
            "type": "number",
            "options": []
        })

for field in categorical_fields:
    options = []

    for column in feature_columns:
        if column.startswith(field + "_"):
            options.append(column[len(field) + 1:])

    if options:
        input_fields.append({
            "name": field,
            "type": "select",
            "options": sorted(options)
        })


HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>CKD Prediction System</title>

    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: Arial, sans-serif;
            background: #f0f7f5;
            color: #253b38;
            min-height: 100vh;
        }

        header {
            background: #286865;
            color: white;
            padding: 22px 7%;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
        }

        .logo {
            font-size: 22px;
            font-weight: bold;
        }

        header span {
            font-size: 13px;
            opacity: 0.9;
        }

        .hero {
            text-align: center;
            padding: 45px 20px 30px;
        }

        .hero h1 {
            color: #286865;
            font-size: 32px;
            margin-bottom: 12px;
        }

        .hero p {
            color: #657875;
            line-height: 1.6;
        }

        .container {
            max-width: 1050px;
            margin: 0 auto 40px;
            padding: 0 20px;
        }

        .form-card {
            background: white;
            padding: 32px;
            border-radius: 16px;
            box-shadow: 0 8px 30px rgba(40, 104, 101, 0.09);
        }

        .section-title {
            color: #286865;
            font-size: 20px;
            margin-bottom: 8px;
        }

        .section-description {
            color: #7b8c88;
            font-size: 13px;
            margin-bottom: 25px;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 22px;
        }

        .form-group label {
            display: block;
            font-size: 14px;
            font-weight: bold;
            margin-bottom: 8px;
            color: #354c47;
        }

        input, select {
            width: 100%;
            padding: 13px;
            border: 1px solid #d7e5e0;
            border-radius: 8px;
            outline: none;
            font-size: 14px;
            background: #fbfdfc;
        }

        input:focus, select:focus {
            border-color: #286865;
            box-shadow: 0 0 0 3px rgba(40, 104, 101, 0.1);
        }

        .predict-btn {
            width: 100%;
            padding: 15px;
            margin-top: 30px;
            border: none;
            border-radius: 9px;
            background: #286865;
            color: white;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: 0.2s;
        }

        .predict-btn:hover {
            background: #1d504d;
        }

        .result {
            margin-top: 25px;
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            background: #e7f5ee;
            border: 1px solid #c6e6d6;
        }

        .result h2 {
            color: #286865;
            margin-bottom: 12px;
        }

        .result h3 {
            font-size: 23px;
            color: #24664b;
            margin-bottom: 10px;
        }

        .error {
            margin-top: 20px;
            padding: 15px;
            color: #a52a2a;
            background: #fff0f0;
            border-radius: 8px;
        }

        footer {
            text-align: center;
            padding: 22px;
            color: #71827e;
            font-size: 13px;
        }

        @media (max-width: 650px) {
            .form-grid {
                grid-template-columns: 1fr;
            }

            .form-card {
                padding: 22px;
            }

            .hero h1 {
                font-size: 25px;
            }
        }
    </style>
</head>

<body>

<header>
    <div class="logo">CKD Prediction System</div>
    <span>Machine Learning Based Healthcare Project</span>
</header>

<section class="hero">
    <h1>Chronic Kidney Disease Prediction</h1>
    <p>
        Enter the patient's medical information below
        to generate a machine learning prediction.
    </p>
</section>

<div class="container">

    <div class="form-card">

        <h2 class="section-title">Patient Information</h2>

        <p class="section-description">
            Please enter all the required medical details.
        </p>

        <form method="POST">

            <div class="form-grid">

                {% for field in fields %}

                <div class="form-group">

                    <label>{{ field.name.upper() }}</label>

                    {% if field.type == "number" %}

                    <input
                        type="number"
                        step="any"
                        name="{{ field.name }}"
                        placeholder="Enter {{ field.name }}"
                        value="{{ request.form.get(field.name, '') }}"
                        required
                    >

                    {% else %}

                    <select name="{{ field.name }}" required>

                        <option value="">Select {{ field.name }}</option>

                        {% for option in field.options %}

                        <option value="{{ option }}"
                            {% if request.form.get(field.name) == option %}
                            selected
                            {% endif %}>
                            {{ option|capitalize }}
                        </option>

                        {% endfor %}

                    </select>

                    {% endif %}

                </div>

                {% endfor %}

            </div>

            <button class="predict-btn" type="submit">
                Predict Kidney Disease
            </button>

        </form>

        {% if result %}

        <div class="result">
            <h2>Prediction Result</h2>
            <h3>{{ result }}</h3>
            <p>{{ message }}</p>
        </div>

        {% endif %}

        {% if error %}
        <div class="error">{{ error }}</div>
        {% endif %}

    </div>

</div>

<footer>
    CKD Prediction System | MCA Machine Learning Project
    <br><br>
    Educational use only. This prediction is not a medical diagnosis.
</footer>

</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    message = None
    error = None

    if request.method == "POST":

        try:
            # Create patient data with all expected model columns
            patient_data = {
                column: 0 for column in feature_columns
            }

            # Read numerical values
            for field in numeric_fields:

                if field not in feature_columns:
                    continue

                value = float(request.form[field])

                patient_data[field] = value

            # Read categorical values
            for field in categorical_fields:

                options = [
                    column[len(field) + 1:]
                    for column in feature_columns
                    if column.startswith(field + "_")
                ]

                if not options:
                    continue

                selected_value = request.form[field]
                column_name = field + "_" + selected_value

                if column_name not in patient_data:
                    raise ValueError(
                        "Invalid selection for " + field
                    )

                patient_data[column_name] = 1

            # Convert patient information into DataFrame
            patient_df = pd.DataFrame(
                [patient_data],
                columns=feature_columns
            )

            # Generate prediction
            prediction = model.predict(patient_df)[0]

            if str(prediction).strip().lower() == "ckd":

                result = "CKD Detected"
                message = (
                    "The model predicts the CKD class. "
                    "Please consult a qualified medical professional."
                )

            elif str(prediction).strip().lower() == "notckd":

                result = "No CKD Detected"
                message = (
                    "The model predicts the not-CKD class. "
                    "This result does not rule out kidney disease."
                )

            else:

                result = "Prediction: " + str(prediction)
                message = "Please consult a medical professional."

        except Exception as e:

            error = "Unable to make prediction: " + str(e)

    return render_template_string(
        HTML_PAGE,
        fields=input_fields,
        result=result,
        message=message,
        error=error
    )


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port
    )
