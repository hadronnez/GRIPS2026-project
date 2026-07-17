

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(
    "C:/Users/adria/Desktop/asuntos_adrian/Temporal_heavy_projects/GRIPS2026-project"
)
csv_path = ROOT / "output" / "stage1" / "hydro_improved_forecast.csv"
out_path = ROOT / "output" / "stage1" / "hydro_residuals.png"

def main(csv_path, out_path="histograma_residuos.png"):
    df = pd.read_csv(csv_path)

    # Residuos = actual - predicción
    resid_forecast = df["actual"] - df["forecast"]
    resid_improved = df["actual"] - df["improved_forecast"]

    # Offset/bias absoluto acumulado (suma total de |residuo|)
    total_abs_forecast = resid_forecast.abs().sum()/len(df["actual"])
    total_abs_improved = resid_improved.abs().sum()/len(df["actual"])

    # Mismo rango de bins para que las dos distribuciones sean comparables
    all_resid = pd.concat([resid_forecast, resid_improved])
    bins = 60
    bin_range = (all_resid.min(), all_resid.max())

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.hist(resid_forecast, bins=bins, range=bin_range, alpha=0.55,
            color="#E15759", label=f"forecast (σ={resid_forecast.std():.4g})",
            edgecolor="white", linewidth=0.3)
    ax.hist(resid_improved, bins=bins, range=bin_range, alpha=0.55,
            color="#4E79A7", label=f"improved_forecast (σ={resid_improved.std():.4g})",
            edgecolor="white", linewidth=0.3)

    ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_xlabel("Residual (Actual - Forecast)")
    ax.set_ylabel("Frequency")
    ax.set_title("Residuals distribution: Forecast vs Improved forecast")
    ax.legend(loc="upper left")

    # Texto con el offset absoluto acumulado de cada modelo
    text = (f"Accumulated ofset normalized:\n"
            f"Forecast: {total_abs_forecast:.4g}\n"
            f"Improved forecast: {total_abs_improved:.4g}")
    ax.text(0.98, 0.97, text, transform=ax.transAxes,
            ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Guardado en {out_path}")
    print(df[["actual", "forecast", "improved_forecast"]].describe())

if __name__ == "__main__":
    main(csv_path, out_path)