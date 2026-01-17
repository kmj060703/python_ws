# src/analysis/analysis_main.py

import os
import argparse

from src.analysis.utils.config import load_config
from src.analysis.io.loaders import load_need_data
from src.analysis.analysis.need_driver import run_need_driver_analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/analysis.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    # paths
    need_path = config["paths"]["need_csv"]
    output_dir = config["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # load
    need_df = load_need_data(need_path)

    # run (Need driver + policy direction)
    run_need_driver_analysis(
        need_df=need_df,
        config=config,
        output_dir=output_dir
    )

    print("✅ Need driver analysis completed.")
    print(f"📁 outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
