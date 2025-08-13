import pandas as pd

# --- 1. Lade die Datensätze ein ---
df1 = pd.read_csv("/Users/ag.saeidi/use_case_3/ground_information.csv")
df2 = pd.read_csv("/Users/ag.saeidi/use_case_3/flight_information.csv")

# --- 2. Konvertiere relevante Datum/Zeit-Spalten ins datetime-Format ---
# Für df1
datetime_columns_df1 = [
    "sched_inbound_dep", "sched_inbound_arr",
    "sched_outbound_dep", "sched_outbound_arr"
]
for col in datetime_columns_df1:
    if col in df1.columns:
        df1[col] = pd.to_datetime(df1[col], errors='coerce')

# Für df2
datetime_columns_df2 = [
    "m_offblockdt", "m_onblockdt"
]
for col in datetime_columns_df2:
    if col in df2.columns:
        df2[col] = pd.to_datetime(df2[col], errors='coerce')

# --- 3. Schlüsselspalten definieren ---
key_columns = ['fn_number', 'ac_registration']

# --- 4. Prüfe, ob die Schlüssel eindeutig sind ---
print("\nSchlüssel in df1:")
print(df1[key_columns].drop_duplicates())

print("\nSchlüssel in df2:")
print(df2[key_columns].drop_duplicates())

# --- 5. Gemeinsame Schlüssel identifizieren ---
common_keys = pd.merge(
    df1[key_columns].drop_duplicates(),
    df2[key_columns].drop_duplicates(),
    on=key_columns
)
print(f"\nGemeinsame Schlüssel (Anzahl: {len(common_keys)}):")
print(common_keys)

# --- 6. Führe die Tabellen über die gemeinsamen Schlüssel zusammen ---
merged_df = pd.merge(df1, df2, on=key_columns, how='inner', suffixes=('_df1', '_df2'))

# --- 7. Zeige die Zusammenfassung ---
print("\nZusammengeführter DataFrame (Vorschau):")
print(merged_df.head())

# Optional: speichere das Ergebnis
merged_df.to_csv("merged_turnaround_crew.csv", index=False)
