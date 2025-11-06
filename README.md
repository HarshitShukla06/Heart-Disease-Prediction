# ❤️ CardioPredictAi – Heart Disease Prediction Using Machine Learning

**CardioPredictAi** is a Machine Learning–based project designed to predict the likelihood of heart disease in patients using **Logistic Regression**.  
By analyzing patient health data such as **age, cholesterol level, blood pressure, and heart rate**, this system provides an early indication of potential cardiovascular risk.  

The model was built using **Python**, trained on the **Heart Disease Prediction Dataset**, and deployed with tools such as **Streamlit** for real-time interaction.

---

## 🚀 Features

- 🧠 **Logistic Regression Model** for binary classification (Heart Disease: Present or Absent)  
- ⚙️ **Automated Data Preprocessing** using `StandardScaler`  
- 📊 **Performance Evaluation** with Accuracy, Recall, F1-Score, and Confusion Matrix  
- 📈 **Data Visualization** using Matplotlib and Seaborn  
- 💾 **Model Persistence** – saves trained model (`.pkl`) files for reuse  
- 🌐 **Deployment-Ready** via Streamlit for real-time prediction  
- 🩺 **Practical Healthcare Utility** – helps clinicians identify risk early  

---

## 📁 Project Structure

CardioPredictAi/
│
├── Heart_Disease_Prediction.csv # Dataset
├── heart_disease_prediction.py # Main ML script (Logistic Regression)
├── heart_disease_prediction_model.pkl # Saved trained model
├── scaler.pkl # Saved StandardScaler object
├── app.py # (Optional) Streamlit app for deployment
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## 🧩 Technologies Used

| Category | Technology / Library | Purpose |
|-----------|----------------------|----------|
| Programming Language | Python | Core development |
| ML Framework | scikit-learn | Model training and evaluation |
| Data Handling | Pandas, NumPy | Data cleaning and computation |
| Visualization | Matplotlib, Seaborn | Graphs, charts, and confusion matrix |
| Model Persistence | joblib | Save and load trained models |
| Deployment (optional) | Streamlit | Interactive web interface for predictions |

---

## ⚙️ Installation & Setup

### **1. Clone the Repository**
```bash
git clone https://github.com/yourusername/CardioPredictAi.git
cd CardioPredictAi
