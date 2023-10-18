#!/bin/bash

# Initialize base source and destination directories relative to the current directory
BASE_SOURCE_DIR="$(pwd)/dataset/"

# # DATASET CONFIG 1 FOR L5 L6
# BASE_DEST_DIR="$(pwd)/dataset_model_1/"
# Rb_VALUES=("1.15")  # Add more Rb values as needed
# DELTA_VALUES=("*")  # Use "*" for any value, or specify values like "0.1", "-0.2", etc.
# BETA_VALUES=("0.5" "1.0" "2.0" "4.0" "8.0" "16.0")  # Add more beta values as needed
# L_VALUES=("5" "6")  # Add more L values as needed

# DATASET CONFIG 2 FOR L5 L6 L11 L12 
# BASE_DEST_DIR="$(pwd)/dataset_model_2/"
# Rb_VALUES=("1.15")  # Add more Rb values as needed
# DELTA_VALUES=("*")  # Use "*" for any value, or specify values like "0.1", "-0.2", etc.
# BETA_VALUES=("0.5" "1.0" "2.0" "4.0" "8.0" "16.0")  # Add more beta values as needed
# L_VALUES=("5" "6" "11" "12")  # Add more L values as needed

# # DATASET CONFIG 3 FOR L5 L6 L11 L12 L15 L16
BASE_DEST_DIR="$(pwd)/dataset_model_3/"
Rb_VALUES=("1.15")  # Add more Rb values as needed
DELTA_VALUES=("*")  # Use "*" for any value, or specify values like "0.1", "-0.2", etc.
BETA_VALUES=("0.5" "1.0" "2.0" "4.0" "8.0" "16.0")  # Add more beta values as needed
L_VALUES=("5" "6" "11" "12" "15" "16")  # Add more L values as needed

# Loop through each L, Rb, delta, and beta value to find and copy directories
for l in "${L_VALUES[@]}"; do
    SOURCE_DIR="${BASE_SOURCE_DIR}L_${l}/"
    DEST_DIR="${BASE_DEST_DIR}L_${l}/"

    # Create the L_X sub-directory in the destination directory if it doesn't exist
    mkdir -p "$DEST_DIR"

    for rb in "${Rb_VALUES[@]}"; do
        for delta in "${DELTA_VALUES[@]}"; do
            for beta in "${BETA_VALUES[@]}"; do
                echo "Searching for directories with L=$l, Rb=$rb, delta=$delta, and beta=$beta"
                find "$SOURCE_DIR" -type d -name "BloqadeQMC_L=${l}_Rb=${rb}_delta=${delta}_beta=${beta}" | xargs -I {} cp -r {} "$DEST_DIR"
            done
        done
    done
done

echo "Done."
