# Install dependencies
!pip install gradio scikit-learn tensorflow pandas xgboost

import gradio as gr
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# -----------------------------
# Load datasets
# -----------------------------
datasets = {
    "Diabetes": {
        "url": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
        "columns": ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin',
                    'BMI','DiabetesPedigreeFunction','Age','Outcome'],
        "target": "Outcome"
    },
    "Heart Disease": {
        "url": "https://raw.githubusercontent.com/plotly/datasets/master/heart.csv",
        "target": "target"
    },
    "Breast Cancer": {
        "url": "https://raw.githubusercontent.com/plotly/datasets/master/data.csv",
        "target": "diagnosis"
    }
}

trained_models = {}

# -----------------------------
# Training function
# -----------------------------
def train_models(disease):
    if disease in trained_models:
        return trained_models[disease]
    
    info = datasets[disease]
    df = pd.read_csv(info["url"])
    
    # Fix column names if needed
    if "columns" in info:
        df.columns = info["columns"]
    
    X = df.drop(columns=[info["target"]])
    y = df[info["target"]]
    
    # Encode target if needed
    if y.dtype == 'object':
        y = y.astype('category').cat.codes
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    models = {}
    
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_scaled, y)
    models["Logistic Regression"] = lr
    
    # SVM
    svm = SVC(probability=True)
    svm.fit(X_scaled, y)
    models["SVM"] = svm
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)
    models["Random Forest"] = rf
    
    # XGBoost
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    xgb_model.fit(X_scaled, y)
    models["XGBoost"] = xgb_model
    
    # Deep Learning
    dl = Sequential([
        Dense(32, activation='relu', input_shape=(X_scaled.shape[1],)),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    dl.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    dl.fit(X_scaled, y, epochs=10, batch_size=16, verbose=0)
    models["Deep Learning"] = dl
    
    trained_models[disease] = (models, scaler, list(X.columns))
    return trained_models[disease]

# -----------------------------
# Prediction function
# -----------------------------
def predict(disease, model_choice, *inputs):
    models, scaler, features = train_models(disease)
    X_input = np.array(inputs).reshape(1, -1)
    X_input_scaled = scaler.transform(X_input)

    model = models[model_choice]
    
    if model_choice == "Deep Learning":
        prob = float(model.predict(X_input_scaled)[0][0])
        pred = 1 if prob >= 0.5 else 0
    else:
        prob = model.predict_proba(X_input_scaled)[0][1]
        pred = model.predict(X_input_scaled)[0]

    # ✅ Human-readable results
    if disease == "Diabetes":
        label = "Diabetic" if pred == 1 else "Non-Diabetic"
    elif disease == "Heart Disease":
        label = "Heart Disease" if pred == 1 else "Healthy"
    elif disease == "Breast Cancer":
        label = "Malignant" if pred == 1 else "Benign"
    else:
        label = str(pred)

    # ✅ Confidence level
    confidence_score = prob * 100
    if confidence_score >= 80:
        confidence_label = f"High Confidence ({confidence_score:.2f}%)"
    elif confidence_score >= 60:
        confidence_label = f"Medium Confidence ({confidence_score:.2f}%)"
    else:
        confidence_label = f"Low Confidence ({confidence_score:.2f}%)"

    return f"{prob:.2f}", label, confidence_label

# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 MediScan: Multi-Model Disease Prediction")
    
    disease = gr.Dropdown(list(datasets.keys()), value="Diabetes", label="Select Disease")
    model_choice = gr.Radio(
        ["Logistic Regression", "SVM", "Random Forest", "XGBoost", "Deep Learning"],
        value="Random Forest",
        label="Select Model"
    )
    
    # Default to Diabetes input fields
    _, _, features = train_models("Diabetes")
    inputs = [gr.Number(label=f) for f in features]
    
    output1 = gr.Textbox(label="Prediction Probability")
    output2 = gr.Textbox(label="Prediction Result (Readable)")
    output3 = gr.Textbox(label="Model Confidence (High/Medium/Low)")
    
    btn = gr.Button("🔍 Predict")
    btn.click(fn=predict, inputs=[disease, model_choice] + inputs, outputs=[output1, output2, output3])

demo.launch(debug=True)
