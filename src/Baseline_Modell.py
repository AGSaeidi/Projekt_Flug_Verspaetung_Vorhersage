from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np

# --- Daten laden ---
X_train = pd.read_csv("/Users/ag.saeidi/Python_Projects/Projects/Fallstudie_Model_Engineering/daten/X_train.csv")
X_test = pd.read_csv("/Users/ag.saeidi/Python_Projects/Projects/Fallstudie_Model_Engineering/daten/X_test.csv")
y_train = pd.read_csv("/Users/ag.saeidi/Python_Projects/Projects/Fallstudie_Model_Engineering/daten/y_train.csv").squeeze()
y_test = pd.read_csv("/Users/ag.saeidi/Python_Projects/Projects/Fallstudie_Model_Engineering/daten/y_test.csv").squeeze()

# --- Baseline-Modell ---
dummy = DummyRegressor(strategy="mean")
dummy.fit(X_train, y_train)

y_pred_dummy = dummy.predict(X_test)

# --- Metriken berechnen ---
mae_dummy = mean_absolute_error(y_test, y_pred_dummy)
rmse_dummy = np.sqrt(mean_squared_error(y_test, y_pred_dummy))
r2_dummy = r2_score(y_test, y_pred_dummy)

# --- Ergebnisse anzeigen ---
print("🔹 Baseline-Modell (DummyRegressor)")
print(f"MAE:  {mae_dummy:.2f}")
print(f"RMSE: {rmse_dummy:.2f}")
print(f"R²:   {r2_dummy:.2f}")
