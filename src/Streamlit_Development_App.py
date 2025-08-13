import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

DATA_PATH = "/Users/ag.saeidi/Python_Projects/Projects/Fallstudie_Model_Engineering/daten/phase3_bereinigt_ml_ready.csv"
OUTPUT_DIR = "output_predictions"
N_ESTIMATORS = 200
RANDOM_STATE = 42

def dep_hour_to_time(dep_hour, mean, std):
    real_hour = dep_hour * std + mean
    real_hour = real_hour % 24  # Clamp auf 0-23 Stunden
    hours = int(real_hour)
    minutes = int((real_hour - hours) * 60)
    minutes = max(0, min(minutes, 59))  # Clamp Minuten auf 0-59
    return f"{hours:02d}:{minutes:02d}"

def inverse_transform_delay(scaled_values, mean, std):
    return scaled_values * std + mean

def train_and_predict(df, tail_col, dep_delay_mean, dep_delay_std, dep_hour_mean, dep_hour_std):
    df_tail = df[df[tail_col] == 1].copy()
    if df_tail.empty:
        return None
    
    # Features ohne Target und Tail-Spalte
    X = df_tail.drop(columns=["dep_delay", tail_col])
    y = df_tail["dep_delay"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    model = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Inverse Skalierung der Verzögerungen
    y_test_min = inverse_transform_delay(y_test, dep_delay_mean, dep_delay_std)
    y_pred_min = inverse_transform_delay(y_pred, dep_delay_mean, dep_delay_std)

    metrics = {
        "MAE": mean_absolute_error(y_test_min, y_pred_min),
        "RMSE": mean_squared_error(y_test_min, y_pred_min) ** 0.5,
        "R2": r2_score(y_test_min, y_pred_min)
    }

    # Vorhersage für alle Zeilen
    day_df = df_tail.copy()
    day_df["Predicted_dep_delay_min"] = inverse_transform_delay(model.predict(X), dep_delay_mean, dep_delay_std)

    # Negative Werte auf 0 setzen
    day_df["Predicted_dep_delay_min"] = day_df["Predicted_dep_delay_min"].clip(lower=0)

    # Zeitformat erzeugen
    day_df["dep_time_str"] = day_df["dep_hour"].apply(lambda x: dep_hour_to_time(x, dep_hour_mean, dep_hour_std))

    # Nur eine Vorhersage pro dep_time_str: mitteln
    day_df = (
        day_df.groupby("dep_time_str", as_index=False)["Predicted_dep_delay_min"]
        .mean()
        .sort_values("dep_time_str")
        .reset_index(drop=True)
    )

    # Kumulative Werte berechnen
    day_df["cumulative_baseline"] = day_df["Predicted_dep_delay_min"].cumsum()
    day_df["cumulative_plus10"] = (day_df["Predicted_dep_delay_min"] * 1.10).cumsum()
    day_df["cumulative_plus30"] = (day_df["Predicted_dep_delay_min"] * 1.30).cumsum()

    return model, metrics, day_df

def plot_cumulative(day_df, tail):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(day_df["dep_time_str"], day_df["cumulative_baseline"], label="Baseline")
    ax.plot(day_df["dep_time_str"], day_df["cumulative_plus10"], label="+10%")
    ax.plot(day_df["dep_time_str"], day_df["cumulative_plus30"], label="+30%")
    plt.xticks(rotation=45)
    plt.xlabel("Abflugzeit")
    plt.ylabel("Kumulative Verspätung (min)")
    plt.title(f"Kumulative Verspätung – {tail}")
    plt.legend()
    plt.tight_layout()
    return fig

def main():
    st.title("Flugverspätung Vorhersage")

    st.info("Daten werden geladen...")
    df = pd.read_csv(DATA_PATH)

    dep_hour_mean = df["dep_hour"].mean()
    dep_hour_std = df["dep_hour"].std()
    dep_delay_mean = df["dep_delay"].mean()
    dep_delay_std = df["dep_delay"].std()

    tail_columns = [col for col in df.columns if col.startswith("ac_registration_")]
    tails = [col.replace("ac_registration_", "") for col in tail_columns]

    selected_tail = st.selectbox("Wähle ein Flugzeug (Tail) zur Vorhersage:", tails)

    if st.button("Vorhersage starten"):
        tail_col = f"ac_registration_{selected_tail}"
        result = train_and_predict(df, tail_col, dep_delay_mean, dep_delay_std, dep_hour_mean, dep_hour_std)
        
        if not result:
            st.warning(f"Keine Daten für {selected_tail}")
            return

        model, metrics, day_df = result

        st.subheader(f"Vorhersage für Tail: {selected_tail}")
        st.write("Metriken:")
        st.json(metrics)

        st.write("Vorhersagedaten (erste 15 Zeilen):")
        st.dataframe(day_df.head(15))

        fig = plot_cumulative(day_df, selected_tail)
        st.pyplot(fig)

        # Optional: speichern
        if st.checkbox("Vorhersagedaten speichern"):
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            tail_dir = os.path.join(OUTPUT_DIR, selected_tail)
            os.makedirs(tail_dir, exist_ok=True)
            day_df.to_csv(os.path.join(tail_dir, "predictions.csv"), index=False)
            st.success(f"Vorhersagedaten gespeichert unter {tail_dir}")

if __name__ == "__main__":
    main()
