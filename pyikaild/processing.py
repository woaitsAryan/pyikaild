"""
This module provides helper functions for generalization and information loss calculation.

It includes functions to generalize categorical and numerical data and to calculate information loss.
"""
import pandas as pd
from typing import List

# Helper function for categorical generalization (simple version)
def generalize_categorical(series: pd.Series) -> str:
    """Generalizes a categorical series to '*' if multiple unique values exist."""
    unique_values = series.unique()
    if len(unique_values) == 1:
        return str(unique_values[0])
    else:
        # In a real scenario with hierarchies, find the lowest common ancestor.
        # For simplicity here, we suppress if not all values are identical.
        return '*'

# Helper function for numerical generalization
def generalize_numeric(series: pd.Series) -> str:
    """Generalizes a numerical series to its range '[min-max]'."""
    min_val = series.min()
    max_val = series.max()
    if min_val == max_val:
        return str(min_val)
    else:
        # You might want formatting options here (e.g., precision)
        return f"[{min_val}-{max_val}]"

# Helper function to calculate Information Loss (simplified)
def calculate_information_loss(
    original_df: pd.DataFrame,
    anonymized_df: pd.DataFrame,
    qi_attributes: List[str],
    numerical_qi: List[str],
    categorical_qi: List[str]
) -> float:
    """
    Calculate a simplified Information Loss metric based on generalization.

    Compare ranges/categories in anonymized vs. original data.

    Note: This is a simplified interpretation of Equation 1. A precise
    implementation requires full domain ranges (max_j, min_j from paper).
    This version uses the actual data min/max as approximations.
    """
    if len(original_df) != len(anonymized_df):
        raise ValueError("Original and anonymized dataframes must have the same length.")

    total_loss = 0.0
    n_records = len(original_df)
    n_qi = len(qi_attributes)

    # Pre-calculate original ranges/uniqueness for normalization
    original_ranges = {}
    for col in numerical_qi:
        min_val = original_df[col].min()
        max_val = original_df[col].max()
        if max_val == min_val:
             original_ranges[col] = 1.0 # Avoid division by zero, range is effectively 1 unit
        else:
            original_ranges[col] = max_val - min_val

    original_cardinality = {}
    for col in categorical_qi:
         # Using number of unique values in the whole dataset as approximation
         # for domain size. A taxonomy height would be better.
        original_cardinality[col] = original_df[col].nunique()
        if original_cardinality[col] == 0: original_cardinality[col] = 1 # Avoid div by zero


    for i in range(n_records):
        record_loss = 0.0
        for col in qi_attributes:
            original_value = original_df.iloc[i][col]
            anonymized_value = anonymized_df.iloc[i][col]

            if pd.isna(anonymized_value) or anonymized_value == '*':
                # Maximum loss for suppression
                loss = 1.0
            elif col in numerical_qi:
                try:
                    # Extract range from '[min-max]' format
                    if isinstance(anonymized_value, str) and '-' in anonymized_value:
                        low, high = map(float, anonymized_value.strip('[]').split('-'))
                        range_width = high - low
                    else: # Value wasn't generalized (single value)
                         range_width = 0.0 # Consider 0 width for single number generalization

                    # Normalize loss by original range
                    if original_ranges[col] > 0:
                         loss = range_width / original_ranges[col]
                    else:
                         loss = 0.0 if range_width == 0 else 1.0 # If original range was 0, any range > 0 is max loss
                except Exception:
                     # Handle cases where conversion fails - treat as max loss
                     # print(f"Warning: Could not parse numeric range '{anonymized_value}' for {col}. Assigning max loss.")
                     loss = 1.0

            elif col in categorical_qi:
                 # Simplified: loss is 0 if value is unchanged, 1 if generalized to '*'
                 # A hierarchy-based approach would calculate loss based on level.
                 loss = 0.0 if str(original_value) == str(anonymized_value) else 1.0
                 # A slightly better categorical loss (normalized by unique values):
                 # unique_in_group = anonymized_df.loc[anonymized_df[col] == anonymized_value, col].nunique() # Expensive way
                 # if str(original_value) == str(anonymized_value):
                 #     loss = 0.0
                 # elif anonymized_value == '*': # Full suppression
                 #     loss = 1.0
                 # else: # Generalized to some category - need hierarchy! Placeholder:
                 #     loss = 1.0 / original_cardinality[col] # Very rough estimate

            else: # Should not happen if QIs are correctly specified
                loss = 0.0

            record_loss += loss

        total_loss += (record_loss / n_qi) # Average loss across QIs for the record

    return total_loss / n_records  # Average loss across all records
