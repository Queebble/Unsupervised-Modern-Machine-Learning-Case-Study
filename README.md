🚀 Unsupervised & Modern Machine Learning Case Study (IFN580)

This project explores a range of unsupervised learning, deep learning, and transformer-based NLP techniques across multiple real-world datasets. It demonstrates practical skills in dimensionality reduction, clustering, time-series forecasting, and modern natural language processing using state-of-the-art architectures such as BERT and T5.

The work is structured into five major components, each focusing on a different machine learning technique or domain.

📂 Project Overview
1. Dimensionality Reduction on Hydrogen Tweet Embeddings

Techniques used: PCA, t-SNE, high-dimensional feature analysis

Preprocessed tweet text using TF-IDF vectorisation.

Demonstrated the curse of dimensionality through distance analysis.

Applied PCA to determine an optimal number of components, analysing explained variance.

Applied t-SNE to visualise embedding structure and compared outcomes with PCA.

2. Clustering on Automotive “Kick” Risk Data

Techniques used: K-Means, feature scaling, cluster evaluation

Preprocessed and standardised numeric vehicle attributes.

Built clustering models to segment vehicles based on odometer, acquisition price, warranty cost, and bad-buy status.

Determined optimal cluster count using elbow and silhouette methods.

Analysed cluster centroids and profiled risk-related vehicle groups.

3. LSTM Forecasting for Agricultural Pasture Growth

Techniques used: LSTM, sequence modelling, multivariate time-series
Dataset: TSDM climate-linked pasture growth measurements

Prepared sequential datasets with fixed lookback windows.

Implemented univariate and multivariate LSTM architectures in TensorFlow/Keras.

Compared training/testing performance using RMSE and R².

Visualised training curves and evaluated model convergence.

4. BERT-Based Hydrogen Tweet Classification

Techniques used: BERT, transformer fine-tuning, binary text classification

Preprocessed hydrogen tweet dataset for transformer input.

Fine-tuned multiple BERT checkpoints for classifying tweets as “relevant” or “irrelevant.”

Evaluated models using accuracy, classification metrics, and attention visualisation.

Compared transformer performance against a logistic regression baseline using TF-IDF features.

5. T5 Question-Answering System on SQuAD-tiny

Techniques used: T5, sequence-to-sequence modelling, ROUGE evaluation
Dataset: Custom subset of SQuAD (1,000 training, 100 validation, 100 test)

Prepared question–context pairs for generative modelling.

Fine-tuned T5 to produce extractive-style answers.

Evaluated performance using ROUGE-1, ROUGE-2, and ROUGE-L scores.

Compared performance against a pre-trained Hugging Face T5 model.

