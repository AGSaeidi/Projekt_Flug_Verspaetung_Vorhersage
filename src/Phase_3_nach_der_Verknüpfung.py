import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- 1. Daten laden ---
df = pd.read_csv("/Users/ag.saeidi/Python_Projects/Projects/Fallstudie_Model_Engineering/merged_turnaround_crew.csv")

# --- 2. Datums-/Zeitspalten kombinieren ---
if 'dep_sched_date' in df.columns and 'dep_sched_time' in df.columns:
    df['scheduled_departure_timestamp'] = pd.to_datetime(df['dep_sched_date'].astype(str) + ' ' + df['dep_sched_time'].astype(str))
else:
    print("Warnung: dep_sched_date oder dep_sched_time fehlt")

if 'arr_sched_date' in df.columns and 'arr_sched_time' in df.columns:
    df['arrival_timestamp'] = pd.to_datetime(df['arr_sched_date'].astype(str) + ' ' + df['arr_sched_time'].astype(str))
else:
    print("Warnung: arr_sched_date oder arr_sched_time fehlt")

# m_offblockdt und m_onblockdt in datetime umwandeln (wenn vorhanden)
for col in ['m_offblockdt', 'm_onblockdt']:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# --- 3. Fehlende Werte behandeln ---
df['ac_registration'] = df['ac_registration'].fillna('UNKNOWN')
df['fn_number'] = df['fn_number'].fillna('UNKNOWN')

# Für numerische Spalten mit fehlenden Werten Median verwenden
num_cols = ['catering_duration', 'cleaning_duration', 'pax_boarding_duration', 'sched_turnaround', 'dep_delay']
for col in num_cols:
    if col in df.columns:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

# --- 4. Inkonsistenzen korrigieren ---
df['fn_number'] = df['fn_number'].str.upper()
df.drop_duplicates(inplace=True)

# --- 5. Feature Engineering ---

# Ground Time in Minuten berechnen (m_offblockdt - m_onblockdt)
if 'm_offblockdt' in df.columns and 'm_onblockdt' in df.columns:
    df['ground_time_min'] = (df['m_offblockdt'] - df['m_onblockdt']).dt.total_seconds() / 60
    df['ground_time_min'] = df['ground_time_min'].clip(lower=0)
else:
    df['ground_time_min'] = 0

# Verzögerungsklassen definieren
def delay_class(x):
    if pd.isna(x) or x <= 5:
        return 'pünktlich'
    elif x <= 30:
        return 'verspätet'
    else:
        return 'stark_verspätet'

if 'dep_delay' in df.columns:
    df['delay_class'] = df['dep_delay'].apply(delay_class)
else:
    df['delay_class'] = 'pünktlich'

# Abflugstunde und Wochentag extrahieren
if 'scheduled_departure_timestamp' in df.columns:
    df['dep_hour'] = df['scheduled_departure_timestamp'].dt.hour
    df['dep_weekday'] = df['scheduled_departure_timestamp'].dt.weekday
else:
    df['dep_hour'] = 0
    df['dep_weekday'] = 0

# --- 6. ML-ready Daten auswählen ---
selected_cols = [
    'ac_registration', 'fn_number', 'delay_class', 'dep_hour', 'dep_weekday',
    'ground_time_min', 'sched_turnaround', 'catering_duration', 'cleaning_duration', 'pax_boarding_duration', 'dep_delay'
]
df_ml = df[selected_cols].copy()

# --- 7. One-Hot-Encoding kategorischer Variablen ---
categorical_cols = ['ac_registration', 'fn_number', 'delay_class']
df_ml = pd.get_dummies(df_ml, columns=categorical_cols, drop_first=True)

# --- 8. Numerische Spalten skalieren ---
numerical_cols = ['ground_time_min', 'sched_turnaround', 'catering_duration', 'cleaning_duration', 'pax_boarding_duration', 'dep_delay', 'dep_hour', 'dep_weekday']
numerical_cols = [col for col in numerical_cols if col in df_ml.columns]

scaler = StandardScaler()
df_ml[numerical_cols] = scaler.fit_transform(df_ml[numerical_cols])

# --- 9. Daten aufteilen ---
target_col = 'dep_delay'
X = df_ml.drop(columns=[target_col])
y = df_ml[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 10. Verzeichnisse prüfen und Daten speichern ---
os.makedirs("daten", exist_ok=True)

df_ml.to_csv("daten/phase3_bereinigt_ml_ready.csv", index=False)
X_train.to_csv("daten/X_train.csv", index=False)
X_test.to_csv("daten/X_test.csv", index=False)
y_train.to_csv("daten/y_train.csv", index=False)
y_test.to_csv("daten/y_test.csv", index=False)

print("Phase 3 abgeschlossen. Daten bereinigt, Features erzeugt, ML-ready Daten erstellt und Split gespeichert.")
