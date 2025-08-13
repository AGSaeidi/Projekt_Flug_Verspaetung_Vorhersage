import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# CSV laden 
df_flights = pd.read_csv("/Users/ag.saeidi/use_case_3/flight_information.csv")

# Zeitspalten verarbeiten
df_flights["scheduled_departure"] = pd.to_datetime(df_flights["dep_sched_date"] + " " + df_flights["dep_sched_time"], errors="coerce")
df_flights["scheduled_arrival"] = pd.to_datetime(df_flights["arr_sched_date"] + " " + df_flights["arr_sched_time"], errors="coerce")
df_flights["off_block_time"] = pd.to_datetime(df_flights["m_offblockdt"], errors="coerce")
df_flights["on_block_time"] = pd.to_datetime(df_flights["m_onblockdt"], errors="coerce")

# Flugzeit berechnen
df_flights["planned_duration"] = df_flights["scheduled_arrival"] - df_flights["scheduled_departure"]
df_flights["actual_duration"] = df_flights["on_block_time"] - df_flights["off_block_time"]
df_flights["planned_duration_min"] = df_flights["planned_duration"].dt.total_seconds() / 60
df_flights["actual_duration_min"] = df_flights["actual_duration"].dt.total_seconds() / 60
# Verteilung vergleichen
sns.histplot(df_flights["planned_duration_min"], label="Geplant", color="blue", kde=True)
sns.histplot(df_flights["actual_duration_min"], label="Reell", color="orange", kde=True)
plt.legend()
plt.title("Geplante vs. tatsächliche Flugzeit")
plt.xlabel("Dauer (Minuten)")
plt.ylabel("Anzahl Flüge")
plt.show()

# Differenz analysieren
df_flights["duration_diff_min"] = df_flights["actual_duration_min"] - df_flights["planned_duration_min"]

# Boxplot zur Übersicht
sns.boxplot(x=df_flights["duration_diff_min"])
plt.title("Abweichung tatsächlicher von geplanter Flugzeit")
plt.xlabel("Differenz (Minuten)")
plt.show()
