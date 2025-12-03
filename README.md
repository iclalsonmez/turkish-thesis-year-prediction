# turkish-thesis-year-prediction

This repository contains the code and experiments for a multi-class **year prediction** task on Turkish academic theses.  
Given the **title** and **abstract** of a thesis, the goal is to predict its **publication year** using ensemble learning methods.

The project was developed as part of the **BLM5109 – Collective Learning** course.


## 1. Problem Definition

- Input:  
  - Turkish thesis **title** (`title_tr`)  
  - Turkish thesis **abstract** (`abstract_tr`)
- Output:  
  - **Year of the thesis** (`year`) as a **multi-class label**

This is formulated as a **multi-class classification problem**, *not* regression.  
Each year (e.g., 2001, 2002, …, 2024) is treated as a separate class, and the models predict:

“Which year class does this thesis belong to?”

Performance is evaluated using classification metrics such as **Accuracy**, **Macro F1**, **Weighted F1**, and **confusion matrices**.


## 2. Dataset

- Source: [umutertugrul/turkish-academic-theses-dataset](https://huggingface.co/datasets/umutertugrul/turkish-academic-theses-dataset) (Hugging Face)
- Language: **Turkish**
- Fields used:
  - `title_tr` – thesis title
  - `abstract_tr` – thesis abstract
  - `year` – thesis year

### Sampling Strategy

- Year range: **2001–2025** (depending on availability after filtering).
- For each year:
   **500 theses** are sampled (if available).
   From those 500:
     **250** are used for **training**
     **250** are used for **testing**
- Total size: `500 × 25 = 12,500` theses (ideal target; some years may have fewer samples, logs show warnings if so).
- Universities and topics are **randomly sampled**; only the year-wise balance is enforced.


## 3. Text Representation (Embeddings)

Thesis titles and abstracts are encoded using a sentence-level embedding model:

- Model: [`ytu-ce-cosmos/turkish-e5-large`](https://huggingface.co/ytu-ce-cosmos/turkish-e5-large)
- Implemented via `SentenceTransformer`

### Embedding Variants

We consider three different input representations:

1. **Title-only embeddings**
   - Shape: `(N, 1024)`
2. **Abstract-only embeddings**
   - Shape: `(N, 1024)`
3. **Combined (Title + Abstract) embeddings**
   - Concatenation of the two:
   - Shape: `(N, 2048)`

Corresponding datasets:

- `title_train_embeddings`, `title_test_embeddings`
- `abstract_train_embeddings`, `abstract_test_embeddings`
- `combined_train_embeddings`, `combined_test_embeddings`

Target labels:

- `y_train`, `y_test` → thesis year (integer labels for 25 classes)


## 4. Models and Ensemble Methods

Three ensemble methods are trained on each representation:

1. **Bagging**
   - Implemented with `BaggingClassifier`
   - Typical tuned hyperparameters:
     - `n_estimators`
     - `max_samples`

2. **Random Subspace**
   - Implemented as Bagging with restricted feature subsets  
     (e.g., controlling `max_features`)
   - Tuned hyperparameters:
     - `n_estimators`
     - `max_features`

3. **Random Forest**
   - Implemented with `RandomForestClassifier`
   - Tuned hyperparameters:
     - `n_estimators`
     - `max_depth`
     - (optionally `max_features`)

For each **(representation - model)** combination, hyperparameters are optimized using **Optuna**, tuning at least **two hyperparameters** per model.


## 5. Evaluation Metrics

For each experiment, the following metrics are computed on the **test set**:

- **Accuracy**
- **Macro F1**
- **Weighted F1**

Short explanation:

- **Macro F1**  
  - Compute F1 **for each class** (each year), then take a simple average.  
  - All years are **equally important** (each year gets 1 vote).
- **Weighted F1**  
  - Compute F1 for each year, then average them **weighted by the number of samples** in each year.  
  - Years with more examples have **more influence**.

In this project, the dataset is constructed to be **approximately balanced by year**, so Macro F1 and Weighted F1 are the same.


## 6. Results

All experiment results are stored in:

- `optimized_results.csv` 

Each row corresponds to a **(representation, model)** pair, with columns such as:

- `Veri Seti` (Representation: `Başlık`, `Özet`, `Birleşik`)
- `Model` (`Bagging`, `Random Forest`, `Random Subspace`)
- `Accuracy`
- `F1-Macro`
- `F1-Weighted`
- HPO details: best hyperparameters, number of trials, paths to saved models/plots

### Main Observations

- **Abstract-based embeddings** (Özet) significantly outperform title-only embeddings.
- **Combined (Title + Abstract)** embeddings provide the strongest representations overall.
- Among ensemble methods:
  - **Random Forest** consistently achieves the best performance,
  - Followed by **Random Subspace**,
  - **Bagging** is generally the weakest.

The **best overall configuration** is:

> **Combined (Title + Abstract) embeddings + Random Forest**  
> (after Optuna hyperparameter optimization)

Test accuracy in this setup is around **8%**, which is noticeably above the random baseline of **1/25 ≈ 4%** for 25 classes.


## 7. Year-wise Analysis

The notebook performs several **year-level analyses** for the best model (Combined + Random Forest):

### 7.1 Confusion Matrix

- A confusion matrix is plotted for all years (classes).
- Patterns:
  - Most errors occur between **neighboring years** (e.g., 2005 vs 2006).
  - There is relatively less confusion between **very distant years** (e.g., 2000 vs 2020).
  - Some years (e.g., “best year” like 2006 in the experiments) show much higher correct counts, while others (e.g., 2002) are systematically harder.

### 7.2 Accuracy per Year

A bar plot is generated where:

- X-axis: year (class)
- Y-axis: accuracy for that year

This highlights:

- Years where the model is particularly strong (high accuracy bars).
- Years where the model struggles (low accuracy bars).

### 7.3 Embedding Visualization (t-SNE)

To better understand why some years are easier or harder, 2D visualizations of the **abstract embeddings** are produced using t-SNE:

- Example comparisons:
  - **Best year (e.g., 2006)** vs **worst year (e.g., 2002)**.
- Observations:
  - Easy years tend to form **more compact, well-separated clusters** in embedding space.
  - Difficult years appear **spread out and overlapping** with other years’ embeddings.

These visualizations support the quantitative findings from the per-year accuracy and confusion matrix.


## 8. Files and Structure (Suggested)

```text
.
├── main.ipynb                # Main notebook
├── optimized_results.csv              # Summary of model performances
├── optimization_plots/                # Optuna optimization history plots
└── README.md
