"""Prepare folder for a labeled Kaggle / custom CSV dataset.



Download a labeled time-series anomaly dataset from Kaggle manually, then place

the CSV at data/kaggle/labeled_series.csv (or update configs/kaggle_labeled.yaml).



Expected columns:

  - one or more numeric feature columns

  - label (0 = normal, 1 = anomaly)

  - optional timestamp column (ignored by default)

"""



from pathlib import Path





def main():

    root = Path(__file__).resolve().parents[1]

    out = root / "data" / "kaggle"

    out.mkdir(parents=True, exist_ok=True)

    readme = out / "README.txt"

    readme.write_text(

        "Place labeled_series.csv here.\n"

        "See configs/kaggle_labeled.yaml for column names.\n",

        encoding="utf-8",

    )

    print(f"Created: {out}")

    print("Add your labeled CSV and run:")

    print("  python experiments/train_ddpm.py --config configs/kaggle_labeled.yaml")





if __name__ == "__main__":

    main()

