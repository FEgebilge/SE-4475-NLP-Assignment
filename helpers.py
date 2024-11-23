import os
import csv

def load_turkish_stop_words_from_csv(csv_path, custom_stop_words=[]):
    csv_path = os.path.abspath(csv_path)  # Resolve absolute path
    stop_words = set(custom_stop_words)

    try:
        with open(csv_path, mode="r", encoding="utf-8") as file:
            reader = csv.reader(file)
            stop_words.update(row[0].strip() for row in reader if row)
    except FileNotFoundError:
        print(f"Error: File {csv_path} not found.")
    except Exception as e:
        print(f"Error reading stop words: {e}")

    return list(stop_words)
