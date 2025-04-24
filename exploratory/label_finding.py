import pandas as pd
from tqdm import tqdm
import requests
import time

# Load the annotations CSV
annotations_path = r"c:\Users\jake3\Desktop\CV Repo\CV-FathomNet2025\data\annotations.csv"
annotations = pd.read_csv(annotations_path)

# Extract unique labels
unique_labels = annotations["label"].unique()

# Define the taxonomic hierarchy order
rank_order = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]

# Function to query full taxonomic data using GBIF API
def get_full_taxonomic_data(label, retries=3, delay=1):
    url = f"https://api.gbif.org/v1/species/match?name={label}"
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                # Print the label and the API response
                print(f"Label: {label}, API Response: {data}")
                resolved_rank = data.get("rank", "Unknown").lower()
                taxonomic_data = {rank: data.get(rank, None) for rank in rank_order}
                
                # Set all ranks below the resolved rank to None
                if resolved_rank in rank_order:
                    rank_index = rank_order.index(resolved_rank)
                    for lower_rank in rank_order[rank_index + 1:]:
                        taxonomic_data[lower_rank] = None
                
                taxonomic_data["rank"] = resolved_rank.capitalize()
                return taxonomic_data
            else:
                print(f"Failed to fetch data for {label}. Status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for {label}: {e}")
        time.sleep(delay)  # Wait before retrying
    return {rank: None for rank in rank_order + ["rank"]}

# Query full taxonomic data for each label
print("Fetching full taxonomic data for all labels...")
taxonomic_data = []
for label in tqdm(unique_labels, desc="Resolving taxonomic data"):
    data = get_full_taxonomic_data(label)
    data["label"] = label  # Add the original label to the data
    taxonomic_data.append(data)

# Create a DataFrame for the taxonomic data
taxonomic_df = pd.DataFrame(taxonomic_data)

# Reorder columns to put 'rank' first
columns = ["rank", "label"] + rank_order
taxonomic_df = taxonomic_df[columns]

# Save the taxonomic data to a CSV file
output_path = r"c:\Users\jake3\Desktop\CV Repo\CV-FathomNet2025\full_taxonomic_data.csv"
taxonomic_df.to_csv(output_path, index=False)
print(f"Full taxonomic data saved to {output_path}")