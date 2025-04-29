# scripts/plot_logs.py

import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_mae_curves(csv_path="results/epoch_logs.csv", save_path="results/mae_per_subject.png"):
    if not os.path.exists(csv_path):
        print(f"Datei nicht gefunden: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    subjects = df["ValidationSubject"].unique()

    plt.figure(figsize=(12, 6))
    for subj in subjects:
        subj_df = df[df["ValidationSubject"] == subj]
        plt.plot(subj_df["Epoch"], subj_df["MAE"], label=f"{subj} - Val", linestyle='-')
        plt.plot(subj_df["Epoch"], subj_df["Train_MAE"], label=f"{subj} - Train", linestyle='--', alpha=0.6)

    plt.xlabel("Epoch")
    plt.ylabel("MAE (cm)")
    plt.title("Trainings- und Validierungs-MAE pro Subjekt")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Plot gespeichert unter: {save_path}")


if __name__ == "__main__":
    plot_mae_curves()
