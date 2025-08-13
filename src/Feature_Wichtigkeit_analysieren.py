import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# --- Daten laden ---
X_train = pd.read_csv("/Users/ag.saeidi/Python_Projects/Projects/Fallstudie_Model_Engineering/daten/X_train.csv")
X_test = pd.read_csv("/Users/ag.saeidi/Python_Projects/Projects/Fallstudie_Model_Engineering/daten/X_test.csv")
y_train = pd.read_csv("/Users/ag.saeidi/Python_Projects/Projects/Fallstudie_Model_Engineering/daten/y_train.csv").squeeze()
y_test = pd.read_csv("/Users/ag.saeidi/Python_Projects/Projects/Fallstudie_Model_Engineering/daten/y_test.csv").squeeze()

# --- Modell trainieren ---
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# --- Feature-Wichtigkeit extrahieren ---
importances = rf.feature_importances_
features = X_train.columns

feat_imp = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# --- Ausgabe der Top-15 Features ---
print("🔹 Wichtigste Einflussfaktoren auf die Ankunftsverspätung:")
print(feat_imp.head(15))

# --- Visualisierung ---
plt.figure(figsize=(10, 6))
plt.barh(feat_imp['Feature'][:15], feat_imp['Importance'][:15])
plt.gca().invert_yaxis()
plt.title("Top 15 Einflussfaktoren auf Ankunftsverspätung")
plt.xlabel("Wichtigkeit")
plt.tight_layout()
plt.show()
