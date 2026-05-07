# Kaggle-Competition-Template
Repo template for Kaggle Competitions
{{Competition-Name}}/
├── data/                   # Git-ignored
│   ├── raw/                # Original data from Kaggle
│   ├── processed/          # Cleaned/engineered features
│   └── external/           # Data from outside sources
├── notebooks/              # Jupyter notebooks
│   ├── eda/                # Exploratory Data Analysis
│   ├── experimental/       # Messy testing/brainstorming
│   └── final_submissions/  # Clean versions of winning kernels
├── src/                    # Modular Python source code
│   ├── __init__.py
│   ├── data_loader.py      # Preprocessing & augmentation
│   ├── models.py           # Model architectures (PyTorch/XGBoost/etc)
│   ├── train.py            # Training loops & validation logic
│   └── utils.py            # Metrics, logging, and helpers
├── models/                 # Git-ignored: Saved .pth, .h5, or .pkl files
├── submissions/            # CSV files ready for upload
├── configs/                # YAML or JSON files for hyperparameters
├── tests/                  # Unit tests for your data pipeline
├── .gitignore              # Crucial: ignore data/ and models/
├── README.md               # Summary, approach, and hardware used
└── requirements.txt        # Exact library versions
