
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Lade den vorbereiteten Datensatz
df = pd.read_csv("/Users/ag.saeidi/Python_Projects/Projects/Fallstudie_Model_Engineering/daten/phase3_bereinigt_ml_ready.csv")

# 2. Filter auf Flugzeug ECLBNX (One-Hot-Encoding!)
df_eclbnx = df[df['ac_registration_ECLBNX'] == 1].copy()

# 3. Zielvariable definieren
target = 'dep_delay'

# 4. Features definieren (alle Spalten außer Ziel)
X = df_eclbnx.drop(columns=[target])
y = df_eclbnx[target]

# 5. Train/Test-Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Modell trainieren
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Vorhersage
y_pred = model.predict(X_test)

# 8. Evaluation (angepasst ohne squared=False)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("📊 Modell-Performance für ECLBNX:")
print(f" - MAE  : {mae:.2f} Minuten")
print(f" - RMSE : {rmse:.2f} Minuten")
print(f" - R²   : {r2:.2f}")

# 9. Visualisierung: Ist vs. Prognose
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.xlabel("Tatsächliche Verspätung (min)")
plt.ylabel("Vorhergesagte Verspätung (min)")
plt.title("ECLBNX: Tatsächliche vs. Vorhergesagte Verspätung")
plt.grid(True)
plt.tight_layout()
plt.show()
