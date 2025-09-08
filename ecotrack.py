#!/usr/bin/env python3
"""
EcoTrack - CLI analysis script
Outputs:
 - co2_plot.png, temp_plot.png, aerosol_plot.png
 - combined_plots.png
 - insights.txt, summary.json
Usage:
  python ecotrack.py            # uses climate_data.csv in same folder
  python ecotrack.py --input myfile.csv --no-show
"""
import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def create_sample_csv(path):
    sample = {
        "Time":["12AM","1AM","2AM","3AM","4AM","6AM","12PM","6PM","12AM_next"],
        "Temperature":[28,27,26,26,27,29,32,30,28],
        "CO2":[410,411,412,413,410,412,414,415,413],
        "Aerosol":[0.1,0.1,0.0,0.0,0.1,0.1,0.2,0.2,0.1]
    }
    df = pd.DataFrame(sample)
    df.to_csv(path, index=False)
    print(f"Sample CSV created at: {path}")

def load_and_clean(path):
    df = pd.read_csv(path)
    # drop fully empty rows
    df = df.dropna(how='all')
    # ensure numeric types
    for col in ['Temperature','CO2','Aerosol']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # simple interpolation for missing numeric values
    df[['Temperature','CO2','Aerosol']] = df[['Temperature','CO2','Aerosol']].interpolate().fillna(method='bfill').fillna(method='ffill')
    # keep original Time order
    df['Time'] = df['Time'].astype(str)
    return df

def plot_series(df, xcol, ycol, outpath, rotate_xticks=True, show=True):
    plt.figure(figsize=(10,5))
    plt.plot(df[xcol], df[ycol], marker='o')
    plt.title(f"{ycol} Over Time")
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.grid(True)
    if rotate_xticks:
        plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(outpath)
    print(f"Saved plot: {outpath}")
    if show:
        plt.show()
    plt.close()

def make_combined_figure(df, outpath, show=True):
    fig, axs = plt.subplots(3,1, figsize=(10,13), constrained_layout=True)
    axs[0].plot(df['Time'], df['CO2'], marker='o'); axs[0].set_title('CO₂ (ppm)'); axs[0].grid(True)
    axs[1].plot(df['Time'], df['Temperature'], marker='s'); axs[1].set_title('Temperature (°C)'); axs[1].grid(True)
    axs[2].plot(df['Time'], df['Aerosol'], marker='^'); axs[2].set_title('Aerosol Index'); axs[2].grid(True)
    for ax in axs:
        for label in ax.get_xticklabels():
            label.set_rotation(30)
    fig.suptitle('EcoTrack — Climate Parameters', fontsize=14)
    fig.savefig(outpath)
    print(f"Saved combined figure: {outpath}")
    if show:
        plt.show()
    plt.close(fig)

def analyze(df):
    n = len(df)
    stats = {
        "n_rows": n,
        "avg_temperature": float(df['Temperature'].mean()),
        "min_temperature": float(df['Temperature'].min()),
        "max_temperature": float(df['Temperature'].max()),
        "avg_co2": float(df['CO2'].mean()),
        "min_co2": float(df['CO2'].min()),
        "max_co2": float(df['CO2'].max()),
        "avg_aerosol": float(df['Aerosol'].mean()),
        "min_aerosol": float(df['Aerosol'].min()),
        "max_aerosol": float(df['Aerosol'].max())
    }
    # detect peaks (time indices with max)
    stats['time_max_temp'] = df.loc[df['Temperature'].idxmax(), 'Time']
    stats['time_max_co2'] = df.loc[df['CO2'].idxmax(), 'Time']
    stats['time_max_aerosol'] = df.loc[df['Aerosol'].idxmax(), 'Time']
    # simple trend check using slope of linear fit vs index
    idx = np.arange(n)
    try:
        stats['temp_trend_slope'] = float(np.polyfit(idx, df['Temperature'].values, 1)[0])
        stats['co2_trend_slope'] = float(np.polyfit(idx, df['CO2'].values, 1)[0])
    except Exception:
        stats['temp_trend_slope'] = None
        stats['co2_trend_slope'] = None
    # generate human-readable insights
    insights = []
    insights.append(f"Average Temperature: {stats['avg_temperature']:.2f} °C (min {stats['min_temperature']}, max {stats['max_temperature']} at {stats['time_max_temp']})")
    insights.append(f"Average CO₂: {stats['avg_co2']:.2f} ppm (min {stats['min_co2']}, max {stats['max_co2']} at {stats['time_max_co2']})")
    insights.append(f"Average Aerosol index: {stats['avg_aerosol']:.3f}")
    # rules / recommendations
    if stats['avg_temperature'] > 30:
        insights.append("⚠ Urban heat concern: average temperature > 30°C. Suggest green cover, reflective roofs, cool pavements.")
    else:
        insights.append("✔ Temperature is in moderate range for sampled period.")
    if stats['avg_co2'] > 412:
        insights.append("⚠ CO₂ levels are high: consider policies like electric public transport and congestion charging.")
    else:
        insights.append("✔ CO₂ levels are moderate in sampled period.")
    if stats['avg_aerosol'] > 0.15:
        insights.append("⚠ Aerosol index is elevated — suggests local pollution sources; recommend monitoring and dust control.")
    return stats, insights

def save_insights_text(insights, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(insights))
    print(f"Saved insights text: {path}")

def save_summary_json(stats, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved summary JSON: {path}")

def main():
    parser = argparse.ArgumentParser(description="EcoTrack: climate analysis")
    parser.add_argument("--input", "-i", default="climate_data.csv", help="CSV input file path")
    parser.add_argument("--no-show", action="store_true", help="Do not call plt.show() (useful on servers)")
    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        create_sample_csv(input_path)
        print("Created sample CSV. Re-run if you want to inspect sample first.")

    df = load_and_clean(input_path)
    show = not args.no_show

    # Plot individual charts
    plot_series(df, 'Time', 'CO2', 'co2_plot.png', rotate_xticks=True, show=show)
    plot_series(df, 'Time', 'Temperature', 'temp_plot.png', rotate_xticks=True, show=show)
    plot_series(df, 'Time', 'Aerosol', 'aerosol_plot.png', rotate_xticks=True, show=show)

    # Combined figure
    make_combined_figure(df, 'combined_plots.png', show=show)

    # Analyze and save outputs
    stats, insights = analyze(df)
    save_insights_text(insights, 'insights.txt')
    save_summary_json(stats, 'summary.json')

    print("\n--- Completed. Files created: co2_plot.png, temp_plot.png, aerosol_plot.png, combined_plots.png, insights.txt, summary.json ---")

if __name__ == "__main__":
    main()
