import pandas as pd

# Lade die Daten 
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
# Sortieren & Gruppieren für Kettenlogik
df_sorted = df_flights.sort_values(by=["ac_registration", "off_block_time"]).copy()

df_sorted["next_off_block"] = df_sorted.groupby("ac_registration")["off_block_time"].shift(-1)
df_sorted["next_legno"] = df_sorted.groupby("ac_registration")["leg_no"].shift(-1)

df_sorted["actual_turnaround_min"] = (
    (df_sorted["next_off_block"] - df_sorted["on_block_time"]).dt.total_seconds() / 60
)

df_sorted["overlap"] = df_sorted["actual_turnaround_min"] < 0
df_sorted["gap_too_long"] = df_sorted["actual_turnaround_min"] > 360
df_sorted["gap_moderate"] = df_sorted["actual_turnaround_min"].between(90, 360)
df_sorted["no_next_flight"] = df_sorted["next_off_block"].isna()

df_problems = df_sorted[df_sorted["overlap"] | df_sorted["gap_too_long"]]

print(f"\n❗️Gefundene Problemfälle in der Kettenlogik: {len(df_problems)}\n")
print(df_problems[[
    "ac_registration", "leg_no", "off_block_time", "on_block_time",
    "next_legno", "next_off_block", "actual_turnaround_min", "overlap", "gap_too_long"
]])
