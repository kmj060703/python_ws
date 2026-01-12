# src/analysis/io/loaders.py

import pandas as pd


def load_need_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required_cols = [
        "district",
        "suicide_rate",
        "depression_experience_rate",
        "perceived_stress_rate",
        "high_risk_drinking_rate",
        "unmet_medical_need_rate",
        "unemployment_rate",
        "elderly_population_rate",
        "old_dependency_ratio",
        "single_households",
        "basic_livelihood_recipients",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[load_need_data] Missing columns in need_tidy.csv: {missing}")

    return df.copy()
