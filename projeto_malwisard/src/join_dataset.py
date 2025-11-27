import numpy as np
import pickle

# =========================================================================================================#
# ====================================== IMPORT DATA ======================================================#
# =========================================================================================================#
print("Importing data...")

# Configuration
preproc = "thermometer12_"
for dataset_type in ["train", "val"]:  # Change to "val" for validation data

    # Define ranges for file chunks
    ranges = [
        (0, 999),
        (1000, 1999),
        (2000, 2999),
        (3000, 3999),
        (4000, 4999),
        (5000, 5999),
        (6000, 6999),
        (7000, 7999),
        (8000, 8999),
        (9000, 9099)
    ] 
    ranges_ = [
        (0, 999),
        (1000, 1999),
        (2000, 2999),
        (3000, 3999),
        (4000, 4999),
        (5000, 5999),
    ]

    if dataset_type == "val":
        ranges = ranges_

    print(f"File prefix: {preproc}{dataset_type}_")
    print("Loading and joining chunks...")

    # Initialize lists for concatenation
    thermometer_X = []
    thermometer_y = []

    # Load and concatenate all chunks
    for start, end in ranges:
        filename_prefix = f"{preproc}{dataset_type}_{start}_{end}"
        
        print(f"Loading {filename_prefix}...")
        X_chunk = pickle.load(open(f"{filename_prefix}X.p", "rb"))
        y_chunk = pickle.load(open(f"{filename_prefix}y.p", "rb"))
        
        thermometer_X.extend(X_chunk)
        thermometer_y.extend(y_chunk if isinstance(y_chunk, list) else list(y_chunk))

    print("Saving combined dataset...")
    output_filename = f"{preproc}{dataset_type}_"
    pickle.dump(thermometer_X, open(f"{output_filename}X.p", "wb"))
    pickle.dump(thermometer_y, open(f"{output_filename}y.p", "wb"))

    print(f"Done! Combined {len(thermometer_X)} samples.")
    print(f"Output files: {output_filename}X.p and {output_filename}y.p")
