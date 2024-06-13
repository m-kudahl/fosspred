<p align="center">
  <img src="https://cdn-icons-png.flaticon.com/512/6295/6295417.png" width="100" />
</p>
<p align="center">
    <h1 align="center">FOSSPred</h1>
    <h2 align="center">Machine-Learning Prediction of ASFI Projects</h2>
</p>
<p align="center">
    <em><code>FOSSPred is a machine learning project aimed at predicting the Graduation & Retirement of Apache Software Foundation Incubator Projects. This repository accompanies the thesis "Predictive Insights: Machine Learning and FOSS Project Sustainability" </code></em>
</p>
<p align="center">
	<img src="https://img.shields.io/github/last-commit/m-kudahl/fosspred?style=flat&logo=git&logoColor=white&color=0080ff" alt="last-commit">
<p>
<p align="center">
		<em>Developed with the software and tools below.</em>
</p>
<p align="center">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white" alt="Python">
</p>
<hr>

## 🔗 Quick Links

> - [📦 Features](#-features)
> - [📂 Repository Structure](#-repository-structure)
> - [🧩 Modules](#-modules)
> - [🚀 Getting Started](#-getting-started)
>   - [⚙️ Installation](#️-installation)
>   - [▶️ Running FOSSPred](#-running-fosspred)


---

## 📦 Features

<code>• Multiple ML-models for prediction</code>

<code>• Evaluation metrics for model performance</code>


---

## 📂 Repository Structure

```sh
└── fosspred/
    ├── README.md
    ├── __init__.py
    ├── the_data.csv
    ├── run.py
    ├── confusion_matrices.txt
    ├── model_results.csv
    ├── models/
    │   ├── decisiontree.py
    │   ├── gradientboosting.py
    │   ├── k-nearest.py
    │   ├── logreg.py
    │   ├── randomforest.py
    │   └── supportvector.py
    └──
```

---

## 🧩 Modules


| File                                                                                              | Summary                         |
| ---                                                                                               | ---                             |
| [run.py](https://github.com/m-kudahl/fosspred/blob/master/run.py)                                 | <code>Main script to run the predictions and write results</code> |
| [confusion_matrices.txt](https://github.com/m-kudahl/fosspred/blob/master/confusion_matrices.txt) | <code>Stores confusion matrices of the models</code> |
| [model_results.csv](https://github.com/m-kudahl/fosspred/blob/master/model_results.csv)           | <code>Logs the results of model evaluations</code> |
| [the_data.csv](https://github.com/m-kudahl/fosspred/blob/master/the_data.csv)                     | <code>Dataset used for training and predictions</code> |



<details closed><summary>models</summary>

| File                                                                                               | Summary                         |
| ---                                                                                                | ---                             |
| [decisiontree.py](https://github.com/m-kudahl/fosspred/blob/master/models/decisiontree.py)         | <code>Decision Tree algorithm</code> |
| [gradientboosting.py](https://github.com/m-kudahl/fosspred/blob/master/models/gradientboosting.py) | <code>Gradient Boosting algorithm</code> |
| [supportvector.py](https://github.com/m-kudahl/fosspred/blob/master/models/supportvector.py)       | <code>Support Vector Machine algorithm</code> |
| [k-nearest.py](https://github.com/m-kudahl/fosspred/blob/master/models/k-nearest.py)               | <code>K-Nearest algorithm</code> |
| [logreg.py](https://github.com/m-kudahl/fosspred/blob/master/models/logreg.py)                     | <code>Logistic Regression algorithm</code> |
| [randomforest.py](https://github.com/m-kudahl/fosspred/blob/master/models/randomforest.py)         | <code>Random Forest algorithm</code> |

</details>

---

## 🚀 Getting Started

***Requirements***

Ensure you have the following installed on your system:

* **Python**: at least `version 3.11.0`

### ⚙️ Installation

1. Clone the fosspred repository:

```sh
git clone https://github.com/m-kudahl/fosspred
```

2. Change to the project directory:

```sh
cd fosspred
```

3. Install the dependencies:

```sh
pip install pandas
pip install scikit-learn
pip install numpy
```

### ▶️ Running FOSSPred

Use the following command to run FOSSPred:

```sh
python run.py
```

---
