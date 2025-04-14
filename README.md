# Uncertainty vs. Robust Explanations

This repository contains the source code used in our study on the relationship between **prediction uncertainty** and the **robustness of explanations**, including both **Counterfactual (cfXp)** and **SHAP (shapXp)** methods.

## 🧠 Introduction

Recent advancements in machine learning have emphasized the need for transparency in model predictions, particularly as interpretability diminishes with increasingly complex architectures. In this work, we propose leveraging **prediction uncertainty** as a complementary approach to classical explainability methods. Specifically, we distinguish between **aleatoric** (data-related) and **epistemic** (model-related) uncertainty to guide the selection of appropriate explanations.

- **Epistemic uncertainty** acts as a rejection criterion for unreliable explanations and provides insight into undertraining.
- **Aleatoric uncertainty** informs the choice between feature-importance and counterfactual explanations.

This framework fosters **uncertainty-aware explainability**, enabling more robust and meaningful interpretations. Our experiments validate the utility of this approach in both classical and deep learning contexts, demonstrating significant gains in the robustness and relevance of explanations.

## 📁 Files Overview

The codebase consists of the following core files:

- `cfXp.py`: Counterfactual explanation implementation  
- `shapXp.py`: SHAP explanation generation  
- `data.py`: Dataset loading and preprocessing  
- `uncertainties.py`: Implementation of uncertainty quantification methods  
- `main.py`: Main script to run experiments

## ⚙️ Set-up

To install all required dependencies, run:

```bash
pip install -r requirements.txt
```

Ensure you are using a Python environment (e.g., `venv` or `conda`) for better reproducibility.

## 🚀 Usage

Run the main script to launch experiments with various combinations of datasets, explanation methods and uncertainty estimation methods:

```bash
python main.py
```

## 📌 Example (IRIS Dataset)

An example using the IRIS dataset is available within the code to help you get started with the structure and usage.

## 📚 References

- A. Hoarau, V. Lemaire, Y. Le Gall, J.-C. Dubois, A. Martin,  
  *Evidential uncertainty sampling strategies for active learning*, Machine Learning, 2024.

- T. Denœux,  
  *A k-nearest neighbor classification rule based on Dempster-Shafer theory*,  
  IEEE Transactions on Systems, Man, and Cybernetics, 1995.

## ⚠️ Note

The code-associated paper is **not yet published**. If you use or reference this code, please make sure to **clearly indicate this status**.

