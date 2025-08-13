import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Flight_Information laden
df_flights = pd.read_csv("/Users/ag.saeidi/use_case_3/flight_information.csv")

# Zeitspalten verarbeiten
df_flights["scheduled_departure"] = pd.to_datetime(df_flights["dep_sched_date"] + " " + df_flights["dep_sched_time"], errors="coerce")
df_flights["scheduled_arrival"] = pd.to_datetime(df_flights["arr_sched_date"] + " " + df_flights["arr_sched_time"], errors="coerce")
df_flights["off_block_time"] = pd.to_datetime(df_flights["m_offblockdt"], errors="coerce")
df_flights["on_block_time"] = pd.to_datetime(df_flights["m_onblockdt"], errors="coerce")

# Flugzeiten berechnen
df_flights["planned_duration"] = df_flights["scheduled_arrival"] - df_flights["scheduled_departure"]
df_flights["actual_duration"] = df_flights["on_block_time"] - df_flights["off_block_time"]
df_flights["planned_duration_min"] = df_flights["planned_duration"].dt.total_seconds() / 60
df_flights["actual_duration_min"] = df_flights["actual_duration"].dt.total_seconds() / 60

# Differenz geplante vs. tatsächliche Flugzeit
df_flights["duration_diff_min"] = df_flights["actual_duration_min"] - df_flights["planned_duration_min"]

# Auswahl numerischer Features aus Flight_Information, die sinnvoll sein könnten
corr_features = [
    "duration_diff_min",
    "dep_delay",          # Abflugverspätung in Minuten
    "Sched Groundtime",   # Geplante Bodenzeit (Minuten)
    "Act Groundtime",     # Tatsächliche Bodenzeit (Minuten)
    "trans_time",         # (z.B. Transferzeit, falls vorhanden)
    "sched_trans_time"    # geplante Transferzeit
]

# Prüfen, welche der Spalten tatsächlich vorhanden sind
corr_features_existing = [col for col in corr_features if col in df_flights.columns]

# Korrelationsmatrix berechnen
corr_matrix = df_flights[corr_features_existing].corr()

# Heatmap der Korrelationen plotten
plt.figure(figsize=(10, 7))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f")
plt.title("Korrelationsmatrix Flight_Information - Einfluss auf Flugzeit-Differenz")
plt.show()
