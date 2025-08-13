import pandas as pd

# 1. DataFrame laden 
df = pd.read_csv('/Users/ag.saeidi/Python_Projects/Projects/Fallstudie_Model_Engineering/merged_turnaround_crew.csv')

# 2. scheduled_departure_timestamp erzeugen (Datum + Uhrzeit)
df['scheduled_departure_timestamp'] = pd.to_datetime(df['dep_sched_date'] + ' ' + df['dep_sched_time'])

# 3. Zeitraum definieren (z.B. 1 Woche)
start_date = '2019-06-01'
end_date = '2019-06-09'

# 4. Filter für Zeitraum (alle Flüge zwischen start_date und end_date)
mask = (df['dep_sched_date'] >= start_date) & (df['dep_sched_date'] <= end_date)
df_period = df.loc[mask].copy()

# 5. Sortieren nach Flugnummer und Abflugzeit (wichtig für Fehlerfortpflanzung)
df_period.sort_values(['fn_number', 'scheduled_departure_timestamp'], inplace=True)

# 6. Vorherige Verspätung pro Flugnummer berechnen
df_period['prev_dep_delay'] = df_period.groupby('fn_number')['dep_delay'].shift(1)

# 7. Gesamt-Korrelation über den Zeitraum berechnen (ohne NaN)
corr_overall = df_period[['dep_delay', 'prev_dep_delay']].dropna().corr().loc['dep_delay', 'prev_dep_delay']
print(f'Gesamtkorrelation der aktuellen Verspätung mit der vorherigen Verspätung von {start_date} bis {end_date}: {corr_overall:.2f}')

# 8. Optional: Korrelation pro Tag berechnen und anzeigen
correlations_per_day = {}
for day in sorted(df_period['dep_sched_date'].unique()):
    df_day = df_period[df_period['dep_sched_date'] == day]
    corr_day = df_day[['dep_delay', 'prev_dep_delay']].dropna().corr()
    if 'dep_delay' in corr_day and 'prev_dep_delay' in corr_day:
        corr_val = corr_day.loc['dep_delay', 'prev_dep_delay']
        correlations_per_day[day] = corr_val
    else:
        correlations_per_day[day] = None

print("\nKorrelationen pro Tag:")
for day, corr_val in correlations_per_day.items():
    if corr_val is not None:
        print(f"{day}: {corr_val:.2f}")
    else:
        print(f"{day}: Nicht berechenbar (zu wenige Daten)")

print(df_period[['fn_number', 'scheduled_departure_timestamp', 'dep_delay', 'prev_dep_delay']])
