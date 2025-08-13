import pandas as pd
import matplotlib.pyplot as plt

# === DATEN EINLESEN ===
df = pd.read_csv("/Users/ag.saeidi/Python_Projects/Projects/Fallstudie_Model_Engineering/merged_turnaround_crew.csv")
ZIEL_FLUGZEUG = "ECLBNX"  
# === DATUMSPARSING ===
df['m_offblockdt'] = pd.to_datetime(df['m_offblockdt'])

# === RELEVANTE SPALTEN REDUZIEREN ===
df = df[['ac_registration', 'm_offblockdt', 'dep_delay']]

# === SORTIEREN & VERSCHIEBUNG ===
df = df.sort_values(by=['ac_registration', 'm_offblockdt'])
df['prev_dep_delay'] = df.groupby('ac_registration')['dep_delay'].shift(1)
df_clean = df.dropna(subset=['prev_dep_delay'])

# === GESAMTKORRELATION FUER ALLE FLUGZEUGE ===
gesamt_corr = df_clean['dep_delay'].corr(df_clean['prev_dep_delay'])
print(f"\nGesamtkorrelation aller Flugzeuge: {gesamt_corr:.2f}")

# === KORRELATION PRO FLUGZEUG ===
flugzeug_korrelationen = df_clean.groupby('ac_registration').apply(
    lambda x: x['dep_delay'].corr(x['prev_dep_delay'])
).dropna()

print("\nTop 10 Korrelationen pro Flugzeug:")
print(flugzeug_korrelationen.sort_values(ascending=False).head(10))

# === HISTOGRAMM DER KORRELATIONEN ===
plt.figure(figsize=(10, 6))
flugzeug_korrelationen.hist(bins=30, edgecolor='black')
plt.title("Verteilung der Korrelationen pro Flugzeug")
plt.xlabel("Korrelation (dep_delay vs. prev_dep_delay)")
plt.ylabel("Anzahl Flugzeuge")
plt.grid(True)
plt.tight_layout()
plt.show()

# === SPEZIFISCHE ANALYSE FUER EIN FLUGZEUG ===
df_ziel = df[df['ac_registration'] == ZIEL_FLUGZEUG].copy()
df_ziel = df_ziel.sort_values(by='m_offblockdt')
df_ziel['prev_dep_delay'] = df_ziel['dep_delay'].shift(1)
df_ziel = df_ziel.dropna()

ziel_corr = df_ziel['dep_delay'].corr(df_ziel['prev_dep_delay'])
print(f"\nKorrelation für Flugzeug {ZIEL_FLUGZEUG}: {ziel_corr:.2f}")

# === ZEITREIHENPLOT FÜR DAS FLUGZEUG ===
plt.figure(figsize=(12, 5))
plt.plot(df_ziel['m_offblockdt'], df_ziel['dep_delay'], label='Aktuelle Verspätung')
plt.plot(df_ziel['m_offblockdt'], df_ziel['prev_dep_delay'], label='Vorherige Verspätung')
plt.title(f"Fehlerfortpflanzung für Flugzeug {ZIEL_FLUGZEUG}")
plt.xlabel("Datum")
plt.ylabel("Verspätung (Minuten)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
