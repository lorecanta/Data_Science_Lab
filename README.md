
# Data Science Lab Project: Financial Literacy Analysis (2017-2020)

## Project Overview
This project focuses on analyzing financial literacy data collected by the Bank of Italy in 2018. The dataset contains detailed information on the knowledge, attitudes, and financial behaviors of Italian citizens.

The primary goal is to perform data transformations and exploratory analysis, with a particular emphasis on discovering association rules through Formal Concept Analysis (FCA).

## Requirements
Before running the project, ensure all necessary Python packages are installed. You can install the dependencies using:

```bash
pip install -r requirements.txt
```

## Project Structure
```bash
├── data/                             # Raw or preprocessed datasets
│   └── Financia_literacy_2018.csv   # Primary dataset from Bank of Italy
│
├── helpers/                          # Reusable utility functions
│   ├── __init__.py
│   ├── utils_analysis.py             # Analysis/statistical methods
│   ├── utils_data_exploration.py     # Data exploration functions
│   └── utils_data_preprocessing.py   # Data cleaning and transformation functions
│
├── Letteratura/                      # Research literature and reference documents
│   ├── Financia literacy - Ricerche.pdf
│   ├── Financial literacy - Ricerche.pdf
│   └── oecd-infe-international.pdf
│
├── output/                           # Output from analysis and visualizations
│   └── output_plots/                 # Automatically generated plots
│
├── pipeline/                         # Data preparation and transformation pipeline
│   ├── __init__.py
│   └── data_preparation.py           # Main data preprocessing script
│
├── column_types.yaml                 # YAML file specifying data types per column
├── config.py                         # Configuration file for paths and constants
├── data_exploration.ipynb            # Notebook for initial data analysis
├── main.ipynb                        # Main notebook: full analysis and FCA application
└── requirements.txt                  # List of required Python libraries

```