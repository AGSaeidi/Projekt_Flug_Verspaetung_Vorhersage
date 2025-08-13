import pandas as pd
df_flights = pd.read_csv("/Users/ag.saeidi/use_case_3/flight_information.csv")
df_turnaround = pd.read_csv("/Users/ag.saeidi/use_case_3/ground_information.csv")
# Zuerst die zusammengesetzten Zeitangaben kombinieren
df_flights["scheduled_departure"] = pd.to_datetime(
    df_flights["dep_sched_date"] + " " + df_flights["dep_sched_time"],
    errors="coerce"
)

df_flights["scheduled_arrival"] = pd.to_datetime(
    df_flights["arr_sched_date"] + " " + df_flights["arr_sched_time"],
    errors="coerce"
)

# Bereits vollständige Zeitspalten umwandeln
df_flights["off_block_time"] = pd.to_datetime(df_flights["m_offblockdt"], errors="coerce")
df_flights["on_block_time"] = pd.to_datetime(df_flights["m_onblockdt"], errors="coerce")

# Überprüfen, ob alles geklappt hat
print(df_flights[["scheduled_departure", "scheduled_arrival", "off_block_time", "on_block_time"]].dtypes)

# Erste Analyse
df_flights.info()
df_flights.describe(include='all')
df_flights.head()
df_flights.nunique()