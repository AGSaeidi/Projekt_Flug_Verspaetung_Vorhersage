import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import norm

# === 1. CSV-Datei laden ===
df = pd.read_csv("/Users/ag.saeidi/Python_Projects/Projects/Fallstudie_Model_Engineering/daten/Phase3_bereinigt_ml_ready_small.csv")

# === 2. Filter auf ECLBNX ===
df_eclbnx = df[df['ac_registration_ECLBNX'] == 1].copy()

# === 3. Sortieren nach dep_hour (Betriebstag-Zeit) ===
df_eclbnx = df_eclbnx.sort_values(by='dep_hour').reset_index(drop=True)

# === 4. Features & Zielvariable definieren ===
target = 'dep_delay'
features = df_eclbnx.drop(columns=[target]).columns
X = df_eclbnx[features]
y = df_eclbnx[target]

# === 5. Train/Test-Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 6. Modell trainieren ===
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === 7. Vorhersage für gesamten Tag ===
y_pred = model.predict(X)

# === 8. Konfidenzintervall berechnen ===
# Annahme: Normalverteilung der Fehler
residuals = y - y_pred
std_dev = np.std(residuals)
confidence_level = 0.95
z_score = norm.ppf((1 + confidence_level) / 2)
ci = z_score * std_dev

# === 9. Fehlerfortpflanzung berechnen ===
cum_error = np.cumsum(np.abs(residuals))

# === 10. Visualisierung ===
plt.figure(figsize=(12, 6))

# A: Prognose + CI
plt.subplot(2, 1, 1)
plt.plot(df_eclbnx['dep_hour'], y, label='Tatsächliche Verspätung')
plt.plot(df_eclbnx['dep_hour'], y_pred, label='Vorhergesagte Verspätung')
plt.fill_between(df_eclbnx['dep_hour'], y_pred - ci, y_pred + ci, color='gray', alpha=0.3, label=f'{int(confidence_level*100)}%-Konfidenzintervall')
plt.title("Vorhersage mit Konfidenzintervall für ECLBNX")
plt.xlabel("Abflugstunde")
plt.ylabel("Verspätung (Minuten)")
plt.legend()
plt.grid(True)

# B: Fehlerfortpflanzung
plt.subplot(2, 1, 2)
plt.plot(df_eclbnx['dep_hour'], cum_error, marker='o', color='red')
plt.title("Fehlerfortpflanzung über den Betriebstag (kumuliert)")
plt.xlabel("Abflugstunde")
plt.ylabel("Kumulierter absoluter Fehler (Minuten)")
plt.grid(True)

plt.tight_layout()
plt.show()
