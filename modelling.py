# Instalasi otomatis (jika belum tersedia)
try:
    import mlflow
except ImportError:
    import os
    os.system('pip install mlflow')

try:
    import pandas as pd
except ImportError:
    os.system('pip install pandas')

try:
    import sklearn
except ImportError:
    os.system('pip install scikit-learn')

# Import library utama
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Aktifkan autolog MLflow
mlflow.sklearn.autolog()

# Load data hasil preprocessing
df = pd.read_csv("dashboard_data_processed.csv")

# Split X dan y
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training model
with mlflow.start_run():
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
