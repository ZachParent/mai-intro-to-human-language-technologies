# Naïve Semantic Similarity

## Overview

This project explores the use of various feature extraction methods and machine learning models to predict the similarity between sentence pairs.

A summary of the results can be found at the bottom of [`train_and_evaluate.ipynb`](./notebooks/train_and_evaluate.ipynb).

## How to Reproduce

Install requirements:

```bash
pip install -r requirements.txt
```

---

Run [`extract_features.ipynb`](./notebooks/extract_features.ipynb) to extract features from the data.

---

Run [`train_and_evaluate.ipynb`](./notebooks/train_and_evaluate.ipynb) to train and evaluate the models.

## Project Structure

- [`src/`](./src/) contains the source code for the project
  - [`zp_ihlt_project/`](./src/zp_ihlt_project/) contains the main code for the project
    - [`feature_extraction.py`](./src/zp_ihlt_project/feature_extraction.py) contains the code for feature extraction
    - [`load_data.py`](./src/zp_ihlt_project/load_data.py) contains the code for loading the data
- [`notebooks/`](./notebooks/) contains the notebooks used for experimentation and reporting
  - [`extract_features.ipynb`](./notebooks/extract_features.ipynb) contains the code for extracting features from the data
  - [`train_and_evaluate.ipynb`](./notebooks/train_and_evaluate.ipynb) contains the code for training and evaluating the models
  - [`explore/`](./notebooks/explore/) contains the historical code used while exploring the data and models (not used in the final report)
- [`reports/`](./reports/) contains the reports for the project
  - [`presentation/`](./reports/presentation/) contains the LaTeX source code for the presentation
    - [`presentation.pdf`](./reports/presentation/presentation.pdf) contains the compiled PDF for the presentation
