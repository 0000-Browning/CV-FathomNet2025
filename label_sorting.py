import pandas as pd
from tqdm import tqdm
import requests
import time

# Load the annotations CSV
annotations_path = r"c:\Users\jake3\Desktop\CV Repo\CV-FathomNet2025\data\annotations.csv"
annotations = pd.read_csv(annotations_path)

# Function to query taxonomic data using GBIF API
def get_taxonomic_data(label, retries=3, delay=1):
    url = f"https://api.gbif.org/v1/species/match?name={label}"
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if "rank" in data and data["rank"]:
                    return data  # Return the full API response
            else:
                print(f"Failed to fetch data for {label}. Status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for {label}: {e}")
        time.sleep(delay)  # Wait before retrying
    return None

# Pre-fetch taxonomic data for all unique labels
print("Fetching taxonomic data for all labels...")
unique_labels = annotations["label"].unique()
taxonomic_data = {}
for label in tqdm(unique_labels, desc="Fetching data"):
    taxonomic_data[label] = get_taxonomic_data(label)

# Function to check if one class is in the hierarchy of another
def is_in_hierarchy(class1, class2):
    data1 = taxonomic_data.get(class1)
    data2 = taxonomic_data.get(class2)

    if not data1 or not data2:
        return False  # If data is missing, assume no hierarchy relationship

    # Extract the rank keys for both classes
    keys1 = {key: data1.get(key) for key in ["kingdomKey", "phylumKey", "classKey", "orderKey", "familyKey", "genusKey", "speciesKey"]}
    keys2 = {key: data2.get(key) for key in ["kingdomKey", "phylumKey", "classKey", "orderKey", "familyKey", "genusKey", "speciesKey"]}

    # Check if keys1 is a subset of keys2 or vice versa
    keys1_set = {k for k, v in keys1.items() if v is not None}
    keys2_set = {k for k, v in keys2.items() if v is not None}

    return keys1_set.issubset(keys2_set) or keys2_set.issubset(keys1_set)

# Compare all classes to find hierarchical subsets
print("Checking for hierarchical subset relationships between classes...")
for i, class1 in enumerate(tqdm(unique_labels, desc="Processing classes")):
    for class2 in unique_labels:
        if class1 != class2 and is_in_hierarchy(class1, class2):
            print(f"Class '{class1}' is in the hierarchy of class '{class2}'")