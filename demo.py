from sklearn.feature_extraction.text import TfidfVectorizer
from zemberek import TurkishMorphology
import pandas as pd
import numpy as np
import re
import os

# Use Zemberek for Turkish Morphology
morphology = TurkishMorphology.create_with_defaults()

def preprocess_text(text):
    # Remove punctuations
    text = re.sub(r"[^\w\s]", "", text)
    
    # Remove URLs, mentions, and numbers
    text = re.sub(r'http\S+|www\S+|@[A-Za-z0-9_]+|\d+', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Convert to lowercase
    text = text.lower()
    
    # Split into words
    words = text.split()
    
    # Filter words (longer than 2 characters and alphabetic)
    words = [w for w in words if len(w) > 2 and w.isalpha()]


    processed_words = []
    for word in words:
        results = morphology.analyze(word)
        if results.analysis_results:
            # If the word is not found, get the root form of the word.
            lemma = results.analysis_results[0].get_stem()
            processed_words.append(lemma)
        else:
            # If the word is found, add the word directly.
            processed_words.append(word)

    return ' '.join(processed_words)

def read_files(document_labels):
    # Define arrays which store the texts and their labels
    texts = []
    labels = []

    # Iterate through the document classes
    for document_label in document_labels:
        print(f"Reading files for document class {document_label}...")
        # Make the full path of the file
        path = f"data/raw_texts/{document_label}"
        for filename in os.listdir(path):
            # Skip non-text files
            if not filename.endswith('.txt'):
                continue
            
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='ISO-8859-9') as file:
                    # Preprocess text
                    texts.append(preprocess_text(file.read()))
                    # Append the document label to the array
                    labels.append(document_label)
    
    # Return the texts and labels which we gathered from files.
    return texts, labels


def create_tfidf_matrix(documents):
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    # Transform documents into TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(documents)
    # Get the feature names
    feature_names = vectorizer.get_feature_names_out()
    # Return the TF-IDF matrix and the features
    return tfidf_matrix, feature_names

def write_tfidf_csv(tfidf_matrix, feature_names):
    # Convert TF-IDF Matrix to Pandas DataFrame
    df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
    # Write the DataFrame into CSV file
    df.to_csv('tfidf_matrix.csv', index=False)


# Calculate cosine similarity between two vecotrs
def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)

    if norm_vector1 == 0 or norm_vector2 == 0:
        return 0.0

    return dot_product / (norm_vector1 * norm_vector2)

def calculate_performance_metrics(actual_labels, predicted_labels):
    classes = np.unique(actual_labels)
    metrics = {}

    for document_class in classes:
        # Reveal the Confusion Matrix
        tp = np.sum((actual_labels == document_class) & (predicted_labels == document_class))
        fp = np.sum((actual_labels != document_class) & (predicted_labels == document_class))
        fn = np.sum((actual_labels == document_class) & (predicted_labels != document_class))

        # Calculate the metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall)

        metrics[document_class] = {
            'True Positives': tp, 
            'False Positives': fp,
            'Recall':  recall,
            'F1-Score': f1_score
        }

    return metrics

def k_fold(X, y):
    # Check whether X is Numpy array or not
    X = X.toarray() if not isinstance(X, np.ndarray) else X

    splits = []
    document_classes = np.unique(y)

    for document_class in document_classes:
        class_indices = np.where(y == document_class)[0]
        np.random.shuffle(class_indices)
        fold_size = len(class_indices) // 10

        for i in range(1, 11):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size # ????

            test_indices = class_indices[test_start:test_end]
            train_indices = np.concatenate([
                class_indices[:test_start],
                class_indices[test_end:]
            ])

            splits.append((
                X[train_indices],
                y[train_indices],
                X[test_indices],
                y[test_indices]
            ))

    return splits
    
# Find k-nearest neighbors using cosine similarity    
def predict(train_data, train_labels, test_point, k):
    similarities = [
        (cosine_similarity(test_point, train_vec), label)
        for train_vec, label in zip(train_data, train_labels)
    ]

    # Sort similarities and select k-nearest neighbors
    similarities.sort(reverse=True)
    k_neighbors = similarities[:k]

    # Select the most occuring class among the k-nearest neighbors
    labels = [label for _, label in k_neighbors]
    return max(set(labels), key=labels.count) if labels else None

def train_and_evaluate(X, y):
    results = {}

    for k in range(1, 11):
        fold_accuracies = []
        fold_metrics = []

        for train_X, train_y, test_X, test_y in k_fold(X, y):
            train_X = np.array(train_X)
            test_X = np.array(test_X)
            predictions = [
                predict(train_X, train_y, test_point, k)
                for test_point in test_X
            ]

            fold_accuracies.append(np.mean(predictions == test_y))
            fold_metrics.append(calculate_performance_metrics(test_y, np.array(predictions)))
            print(f"K: {k}, Accuracy: {fold_accuracies[-1]}")

        avg_accuracy = np.mean(fold_accuracies)
        results[k] = {
            "avg_accuracy": avg_accuracy,
            "fold_metrics": fold_metrics
        }
    
    return results

def get_best_k(results):
    best_k = max(results, key=lambda k: results[k]["avg_accuracy"])
    best_results = {
        "best_k": best_k,
        "best_accuracy": results[best_k]["avg_accuracy"],
        "detailed_metrics": results[best_k]["fold_metrics"]
    }

    return best_results

def main():
    texts, labels = read_files([1, 2, 3])
    
    print("Number of Documents:", len(texts))
    print("Number of Labels:", len(labels))
    print("Creating TF-IDF Matrix...")
    tfidf_matrix, feature_names = create_tfidf_matrix(texts)

    print("Writing TF-IDF Matrix into CSV...")
    write_tfidf_csv(tfidf_matrix, feature_names)



    X = tfidf_matrix.toarray()
    y = np.array(labels)
    print("Training and Evaluating...")
    results = train_and_evaluate(X, y)

    print("Results:")
    best_result = get_best_k(results)


    print("Best K:", best_result["best_k"])
    print("Best Accuracy:", best_result["best_accuracy"])


if __name__ == "__main__":
    main()

