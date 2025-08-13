import matplotlib.pyplot as plt
import numpy as np

# Beispielhafte Legs (1..n)
legs = np.arange(1, 7)

# Median-Verspätungen aus der Szenario-Analyse
baseline_median = [0.5, 0.4, 0.3, 0.5, 0.6, 0.4]
plus10_median   = [10.4, 10.8, 11.2, 11.5, 11.8, 12.0]
plus30_median   = [30.5, 31.8, 33.2, 34.0, 34.8, 35.5]

# Kumulative Summe der Verspätungen
baseline_cum = np.cumsum(baseline_median)
plus10_cum   = np.cumsum(plus10_median)
plus30_cum   = np.cumsum(plus30_median)

# Plot erstellen
plt.figure(figsize=(12,6))

plt.plot(legs, baseline_cum, marker='o', label='Baseline', color='blue')
plt.plot(legs, plus10_cum, marker='o', label='+10 min Startdelay', color='orange')
plt.plot(legs, plus30_cum, marker='o', label='+30 min Startdelay', color='red')

plt.title("Szenario-Analyse: Kumulative Verspätungsentwicklung über den Betriebstag")
plt.xlabel("Leg-Nummer im Tagesverlauf")
plt.ylabel("Kumulative Verspätung (Minuten)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
