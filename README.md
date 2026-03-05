# Interpretable Machine Learning for 2D Materials Property Prediction

This repository implements a **Gradient-Boosted Feature Selection (GBFS)** driven machine learning pipeline for predicting key physical properties of **two-dimensional (2D) inorganic materials**. The workflow focuses on producing **accurate and fully interpretable models** through systematic feature selection followed by model interpretation using SHAP analysis.

Two-dimensional (2D) inorganic crystals are a class of materials that are gaining significant attention for use in electronic and optoelectronic devices. Among many exciting applications, 2D materials offer a range of beneficial properties due to **charge carrier confinement, high carrier mobility, tunable band gaps, strong light–matter interactions, and atomically thin geometries** that enable excellent electrostatic control and mechanical flexibility.

In parallel, **data-driven approaches to predicting inorganic material properties** have gained considerable attention as computationally lightweight surrogate models for expensive first-principles simulations. These models are particularly valuable for **high-throughput screening of candidate materials** when searching for specific structure–property relationships. However, many existing approaches focus primarily on **three-dimensional (3D) bulk crystalline materials**.

In this work we develop **data-driven models for predicting properties of 2D layered, van der Waals, and ultra-thin film materials**, including:

- **Thermodynamic stability**
- **Metallicity**
- **Electronic band gap**

Training data are sourced from several open computational databases of 2D materials:

- `'alexandria_2d'`
- `'c2db'`
- `'mc2d'`
- `'2dmatpedia'`

Chemically meaningful **elemental, physical, and compositional descriptors** are used as model inputs. Because the resulting feature space is large, we apply a **Gradient-Boosted Feature Selection (GBFS)** strategy to identify the most informative descriptors before final model training.

The resulting models are **fully interpretable**, using:

- Feature importance from gradient boosting
- **SHapley Additive exPlanations (SHAP)** for global and local interpretation

---

# Repository Workflow

The repository is organized as a **three-stage pipeline**:

1. **Dataset construction**
2. **Gradient-Boosted Feature Selection (GBFS) and model training**
3. **SHAP interpretation and post-processing of GBFS results**

Each stage is implemented as a Jupyter notebook.

---

# Step 1 — Combine Feature Files

Run the notebook:

- `pkl/combine_features_files.ipynb`

This notebook merges individual feature files into a **single combined dataset** used for the GBFS workflow.

The output is a features .pkl file:

`2d_combined_features.pkl`


This file will serve as the input to the modelling pipeline.

---

# Step 2 — Run Gradient-Boosted Feature Selection and Model Training

Run the notebook:

- `run_all_scripts.ipynb`

This notebook performs the core modelling workflow and **implements the Gradient-Boosted Feature Selection (GBFS) workflow**.
The flags and configurations are listed below.

### Main tasks performed

- Load the combined dataset
- Apply dataset filtering
- Remove unwanted features
- Run **Gradient-Boosted Feature Selection (GBFS)** workflow
- Train the final predictive model
- Evaluate model performance

### Why GBFS?

The initial feature set contains many potentially redundant or weak predictors.  
**GBFS uses gradient-boosted decision trees to iteratively identify the most informative features**, producing:

- A **compact subset of predictive descriptors**
- **Reduced overfitting**
- **Improved interpretability**
- **Better generalization**

The selected feature subset is then used for final model training.

---
# Step 3 — Compare to experimental values in pkl/expt_band_gap folder

Run the notebook:
- `process_expt_dft_gbfs.ipynb`

This notebook performs:

- Processing of **experimental band gap data**
- Comparison between DFT values and GBFS predictions
- Evaluation of model performance **outside the training distribution**
- **Generation of comparison plots and performance metrics**

---

# Step 4 — SHAP Analysis and Re-plotting for Publication

Run the notebooks:
- `process_feature_statistics.ipynb`
- `process_training_figures_classification.ipynb`
- `process_training_figures_regression.ipynb`
- `process_performance_figures_classification.ipynb`
- `process_performance_figures_regression.ipynb`


After training the models, the final stage performs **SHAP analysis and post-processing of GBFS results**.

This stage includes:

- **SHAP analysis** to determine global and local feature contributions
- Visualization of **feature importance rankings**
- Interpretation of **how feature values influence predictions**
- Analysis of the **GBFS-selected feature set**

These analyses allow the resulting models to be **fully explainable**, revealing which physical descriptors most strongly influence predicted material properties.

---

# Configuration Flags (run_all_scripts.ipynb)
The behaviour of the GBFS pipeline is controlled by several configuration parameters defined at the beginning of `run_all_scripts.ipynb`.

---

## Data Path
```python
file_path = os.getcwd()
```
### Description
Path to the directory containing the dataset.

### Default
Uses the current working directory.

## Dataset Name
```python
file_name = '2d_combined'
```
### Description
Base name of the initial file. There is a **seperate features** (`fr"{file_name}_features.pkl"`) and **data file** (`fr"{file_name}_data.pkl"`).

---

## Target Property
```python
target = 'bandgap'
```

### Description
Defines the property to be predicted.

### Options
`'bandgap'`
`'is_stable'`
`'is_metal'`

---

## Objective Type
```python
objective = 'regression'
```
### Description
Defines the machine learning task type.
Default is predicting continuous band gap values.

### Options

| Target | Task Type |
|--------|-----------|
|bandgap | regression |
| is_stable	| classification |
| is_metal | classification |

---

## Unit
```python
unit = 'eV'
```

### Description
Specifies the unit of the target variable.

### Options
`"eV"`
`None`

---

## Boosting Method
```python
boosting_method = 'lightGBM'
```

### Description
Defines the gradient boosting algorithm used throughout GBFS workflow

### Options
`'lightGBM'`
`'XGBoost'`

---

# Filtering Flags (run_all_scripts.ipynb)
These parameters allow the dataset to be filtered before GBFS and model training.

---

## Target Filter
`target_filter`


### Description
Removes materials with specified property value ranges (applicable to `'regression'` tasks only)

### Options
`None`
`Custom lambda function`

### Example:
```python
target_filter = lambda x: x > 0.2
```
---

## Dataset Filter
`dataset_filter`

### Description
Allows manual removal of specific dataset entries prior to training.

### Options

- `None`
- `'alexandria_2d'`
- `'c2db'`
- `'mc2d'`
- `'2dmatpedia'`

### Example:
```python
dataset_filter = 'mc2d'
```

---

## Feature Filter
`features_filter`

### Description
Removes selected features before GBFS workflow is executed.

### Options
- `None`
- `List of feature names`

### Example:
```python
features_filter = ['feature_1', 'feature_2']
```

---

## Physical Property Filters

These options filter the dataset based on known physical classifications from the **2d_combined_data.pkl** file.

### Stability Filter
```python
stability_filter = False
```
If True, only thermodynamically stable materials are retained.

### Metallicity Filter
```python
metallicity_filter = False
```
If True, the dataset will be filtered according to metallicity classification, **retaining only non-metals** in the training/test set for GBFS

---
# Recommended Usage (tl;dr)

## Typical workflow:

### Combine feature datasets
```python
pkl/combine_features_files.ipynb
```

### Run GBFS feature selection and model training
```python
run_all_scripts.ipynb
```

### Compare to experimental dataset
```python
process_compare_expt_dft_gbfs.ipynb
```

### Perform SHAP interpretation and GBFS post-processing
```python
process_feature_statistics.ipynb
process_training_figures_classification.ipynb
process_training_figures_regression.ipynb
process_performance_figures_classification.ipynb
process_performance_figures_regression.ipynb
```

---
# Citation

If you use this repository in your research, please cite the associated publication: 
