# Turkish Multi-Class Text Classification using k-NN and Zemberek

This repository contains an NLP project that classifies Turkish tweets into three categories: Positive, Negative, and Neutral. The project demonstrates text preprocessing, feature extraction, and classification techniques tailored to Turkish language intricacies.

## Features
- Tokenization and stemming of Turkish text using Zemberek NLP library.
- Transformation of text data into TF-IDF vectors for machine learning.
- Multi-class classification using k-Nearest Neighbors (k-NN) with cosine similarity.
- Evaluation using stratified 10-fold cross-validation.
- Comprehensive reporting of performance metrics.

## Dataset
The dataset consists of 3000 Turkish tweets divided into three folders:
- **Positive Tweets**: 756 samples
- **Negative Tweets**: 1287 samples
- **Neutral Tweets**: 957 samples

The dataset is structured as follows:
```
Raw_texts/
    1/
    2/
    3/
```

## Requirements
### Python Libraries
- `numpy`
- `pandas`
- `sklearn`
- `py4j`
- `jpype1`
- `matplotlib`

### Other Requirements
- Zemberek NLP library (`zemberek-full.jar`) for Turkish text processing.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/turkish-text-classification.git
   cd turkish-text-classification
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download **Zemberek**:
   - Visit the [Zemberek GitHub Releases](https://github.com/ahmetaa/zemberek-nlp/releases).
   - Download the latest `zemberek-full.jar` file and save it in the project root.

4. Install the `py4j` library for Python-Java integration:
   ```bash
   pip install py4j
   ```

## Usage
### Preprocessing
- Tokenize and stem Turkish text using Zemberek.
- Convert text to lowercase and optionally remove stop words.

### Feature Representation
- Compute TF-IDF values for all words in the dataset.
- Save the TF-IDF matrix in CSV format.

### Classification
- Use k-NN with cosine similarity to classify tweets.
- Perform stratified 10-fold cross-validation to evaluate the model.

### Performance Metrics
- Precision
- Recall
- F-Score
- Macro and Micro averages

### Steps to Run
1. **Start Zemberek**:
   Launch the Zemberek gateway using the provided script:
   ```python
   from py4j.java_gateway import launch_gateway
   launch_gateway(classpath="zemberek-full.jar")
   ```

2. **Run Preprocessing**:
   Open the Jupyter Notebook `preprocessing.ipynb` and execute the cells to tokenize, stem, and process the tweets.

3. **TF-IDF Conversion**:
   Execute the `tfidf.ipynb` notebook to transform the text into numerical features.

4. **Classification**:
   Use the `classification.ipynb` notebook to run k-NN classification and generate performance reports.

5. **Generate Reports**:
   Save the evaluation metrics and TF-IDF matrix as CSV files.

## Project Structure
```
SE-4475-NLP-Assignment/
├── data/
│   ├── raw_texts/
│   │   ├── 1/
│   │   ├── 2/
│   │   ├── 3/
│   │   ├── stop_words.csv
├── zemberek-full.jar
├── notebooks/
│   ├── preprocessing.ipynb
│   ├── tfidf.ipynb
│   ├── classification.ipynb
├── reports/
│   ├── tfidf_values.csv
│   ├── performance_metrics.csv
├── helpers.py
├── requirements.txt
└── README.md
```

## Evaluation
The k-NN classifier will be tested for various `k` values (e.g., 1, 3, 5) using cosine similarity and other distance metrics like Euclidean. Results will include:
- Precision, Recall, and F-Score per class.
- Macro and Micro averages.

## References
- [Zemberek NLP Library](https://github.com/ahmetaa/zemberek-nlp)
- Python libraries: `numpy`, `pandas`, `sklearn`, `py4j`

---

## Notes
Ensure the Zemberek JAR file is placed in the project root, and the `py4j` gateway is running before executing any notebooks.

---

## License
This project is licensed under the MIT License. See `LICENSE` for details.
