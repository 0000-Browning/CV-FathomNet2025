import pandas as pd
from tqdm import tqdm
import requests
import time

# Load the annotations CSV
annotations_path = r"c:\Users\jake3\Desktop\CV Repo\CV-FathomNet2025\data\annotations.csv"
annotations = pd.read_csv(annotations_path)

# Extract unique labels
unique_labels = annotations["label"].unique()

# Function to query taxonomic rank using GBIF API
def get_taxonomic_rank(label, retries=3, delay=1):
    url = f"https://api.gbif.org/v1/species/match?name={label}"
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                # Print the label and the API response
                print(f"Label: {label}, API Response: {data}")
                if "rank" in data and data["rank"]:
                    return data["rank"]
            else:
                print(f"Failed to fetch rank for {label}. Status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Error fetching rank for {label}: {e}")
        time.sleep(delay)  # Wait before retrying
    return "Unknown"

# Query taxonomic ranks for each label
taxonomic_ranks = {}
for label in tqdm(unique_labels, desc="Resolving taxonomic ranks"):
    temp =get_taxonomic_rank(label)
    taxonomic_ranks[label] =temp 
    print(f"OUTPUT: {label}: {temp}")

# Create a DataFrame for unique labels and their ranks
labels_with_ranks = pd.DataFrame({
    "Label": list(taxonomic_ranks.keys()),
    "Rank": list(taxonomic_ranks.values())
})

# Define a custom order for taxonomic ranks (from high to low)
rank_order = [
    "KINGDOM", "PHYLUM", "CLASS", "ORDER", "FAMILY", "GENUS", "SPECIES", "UNKNOWN"
]

# Sort the DataFrame by rank
labels_with_ranks["Rank"] = pd.Categorical(labels_with_ranks["Rank"], categories=rank_order, ordered=True)
labels_with_ranks = labels_with_ranks.sort_values(by="Rank")

# Print the sorted labels and ranks
print(labels_with_ranks)

# Save the sorted labels and ranks to a CSV file
output_path = r"c:\Users\jake3\Desktop\CV Repo\CV-FathomNet2025\sorted_labels_with_ranks.csv"
labels_with_ranks.to_csv(output_path, index=False)
print(f"Sorted labels with ranks saved to {output_path}")