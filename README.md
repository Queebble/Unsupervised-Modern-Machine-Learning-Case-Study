Absolutely — here is your **Assignment 2 README rewritten in the *exact same format and style* as your Rainfall & SOI Regression Analysis README**.
This uses identical headings, spacing, tone, and structure so your GitHub portfolio looks consistent.

Everything is plain Markdown (ASCII-safe) so it will paste perfectly.

---

# Unsupervised & Modern Machine Learning Case Study (IFN580)

This project applies a range of **unsupervised learning**, **deep learning**, and **transformer-based NLP techniques** to multiple real-world datasets.
It demonstrates skills in dimensionality reduction, clustering, time-series forecasting, and modern NLP using architectures such as **LSTM**, **BERT**, and **T5**.

## 📊 Project Goals

* Reduce high-dimensional tweet embeddings using PCA and t-SNE
* Cluster automotive "kick" risk data using K-Means
* Build univariate and multivariate LSTM models for time-series forecasting
* Fine-tune BERT for tweet relevance classification
* Train a T5 model for a question-answering system using a SQuAD subset
* Evaluate models using metrics such as silhouette score, RMSE, R2, ROC-AUC, and ROUGE

## 🛠️ Tools & Libraries

* Python
* NumPy, Pandas
* Scikit-learn
* TensorFlow / Keras
* Hugging Face Transformers
* Matplotlib, Seaborn
* Jupyter Notebook

## 📈 Key Components

### **1. Dimensionality Reduction (Hydrogen Tweets)**

* Preprocessed tweets using TF-IDF features
* Demonstrated the curse of dimensionality
* Applied PCA to determine optimal components
* Used t-SNE to visualise non-linear structure
* Compared PCA vs t-SNE outputs

### **2. Clustering (Automotive Kick Risk Data)**

* Standardised numeric features
* Built K-Means clustering models
* Selected optimal K via elbow and silhouette methods
* Analysed cluster centroids and kick-risk characteristics

### **3. LSTM Time-Series Forecasting (Agricultural Pasture Growth)**

* Created fixed-length sequences for univariate and multivariate inputs
* Implemented multiple LSTM architectures
* Evaluated performance using RMSE and R2
* Visualised loss curves for convergence analysis

### **4. BERT Classification (Hydrogen Tweets)**

* Prepared text for transformer input
* Fine-tuned BERT models for binary classification
* Compared performance against logistic regression baseline
* Analysed classification behaviour via attention weights

### **5. T5 Question-Answering System (SQuAD Tiny)**

* Fine-tuned T5 on question-context pairs
* Evaluated generated answers using ROUGE metrics
* Compared results with a pre-trained T5 model
* Analysed differences in accuracy and fluency

## 📁 Files

* `Assignment2.ipynb`: Full notebook containing all models and analyses
* Dataset files provided through coursework (not included in repository)

## 🧠 Key Insights

* PCA captures global structure, while t-SNE reveals finer-grained clusters
* K-Means identifies meaningful risk-related groupings in automotive data
* Multivariate LSTM models outperform univariate baselines for pasture growth forecasting
* BERT significantly outperforms traditional TF-IDF + logistic regression approaches
* T5 produces coherent answers even with limited training data, but lags behind fully pretrained models

## 🔄 Reproducibility

To run this project:

1. Clone the repository
2. Open the notebook in Jupyter
3. Install dependencies (`pip install -r requirements.txt` if provided)
4. Run each section sequentially

---

Created as part of a machine learning coursework project, demonstrating skills in modern ML, NLP, time-series modelling, and unsupervised learning.

---
