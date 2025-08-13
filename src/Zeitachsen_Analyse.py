
import pandas as pd

# CSV einlesen
df_flights = pd.read_csv("/Users/ag.saeidi/use_case_3/flight_information.csv")

# Zeitspalten korrekt kombinieren und umwandeln
df_flights["scheduled_departure"] = pd.to_datetime(df_flights["dep_sched_date"] + " " + df_flights["dep_sched_time"])
df_flights["scheduled_arrival"] = pd.to_datetime(df_flights["arr_sched_date"] + " " + df_flights["arr_sched_time"])
df_flights["off_block_time"] = pd.to_datetime(df_flights["m_offblockdt"])
df_flights["on_block_time"] = pd.to_datetime(df_flights["m_onblockdt"])

# Dauer berechnen
df_flights["planned_duration"] = df_flights["scheduled_arrival"] - df_flights["scheduled_departure"]
df_flights["actual_duration"] = df_flights["on_block_time"] - df_flights["off_block_time"]

# In Minuten umrechnen
df_flights["planned_duration_min"] = df_flights["planned_duration"].dt.total_seconds() / 60
df_flights["actual_duration_min"] = df_flights["actual_duration"].dt.total_seconds() / 60
df_flights[["scheduled_departure", "scheduled_arrival", "off_block_time", "on_block_time", "planned_duration_min", "actual_duration_min"]].head()
