import pandas as pd
df_flights = pd.read_csv("/Users/ag.saeidi/use_case_3/flight_informatio\
n.csv")
# Überblick über fehlende Werte
missing_values = df_flights.isnull().sum()
missing_percent = (missing_values / len(df_flights)) * 100
print(pd.DataFrame({"Fehlende Werte": missing_values, "%": missing_percent}))
