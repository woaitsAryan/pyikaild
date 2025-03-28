"""
This module implements the Improved k-Anonymization (IKA) algorithm.

The IKA algorithm uses generalization techniques to ensure k-anonymity in datasets.
"""
import pandas as pd
from typing import List, Dict, Optional, Any
from pyikaild.processing import (
    generalize_categorical,
    calculate_information_loss,
    generalize_numeric
)


class IKA:
    """
    Implement Improved k-Anonymization (IKA) using generalization.

    Uses a recursive partitioning approach inspired by Mondrian to divide
    the dataset into equivalence classes (partitions) where each class
    contains at least 'k' records and records within a class are indistinguishable
    based on the generalized Quasi-Identifier (QI) attributes.

    Attributes:
        k (int): The minimum size of an equivalence class.
        qi_attributes (List[str]): List of column names to be treated as Quasi-Identifiers.
        sa_attribute (str): Column name of the Sensitive Attribute. (Used for context, not anonymized here).
        numerical_qi (List[str]): List of QI attributes that are numerical.
        categorical_qi (List[str]): List of QI attributes that are categorical.
        max_split_level (int): Maximum recursion depth for splitting (controls granularity). Default 10.
        anonymized_data (Optional[pd.DataFrame]): Stores the result after transform.
        partitions (List[pd.Index]): Stores the indices of records belonging to each final partition.
        generalization_map (Dict): Stores the generalized values for each partition and QI.
    """

    def __init__(self,
                 k: int,
                 qi_attributes: List[str],
                 sa_attribute: str, # Keep for context, consistency with paper
                 numerical_qi: Optional[List[str]] = None,
                 categorical_qi: Optional[List[str]] = None,
                 max_split_level: int = 10):
        """Initialize the IKA class with the given parameters."""
        if k < 2:
            raise ValueError("k must be at least 2.")
        if not qi_attributes:
            raise ValueError("qi_attributes list cannot be empty.")
        if not sa_attribute:
             raise ValueError("sa_attribute must be specified.")

        self.k = k
        self.qi_attributes = qi_attributes
        self.sa_attribute = sa_attribute # Primarily for context/downstream use (like ILD)
        self.max_split_level = max_split_level

        # Auto-detect numerical/categorical if not provided (simple heuristic)
        self._numerical_qi = numerical_qi if numerical_qi is not None else []
        self._categorical_qi = categorical_qi if categorical_qi is not None else []
        self._auto_detect_types_needed = not (numerical_qi or categorical_qi)

        self.anonymized_data: Optional[pd.DataFrame] = None
        self.partitions: List[pd.Index] = []
        self.generalization_map: List[Dict[str, Any]] = []
        self._original_df_for_loss: Optional[pd.DataFrame] = None # Store original for loss calc

    def _validate_and_prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check columns, handle type detection, and return a working copy."""
        df_copy = df.copy()
        missing_qi = [col for col in self.qi_attributes if col not in df_copy.columns]
        if missing_qi:
            raise ValueError(f"Missing QI attributes in DataFrame: {missing_qi}")
        if self.sa_attribute not in df_copy.columns:
             raise ValueError(f"Missing SA attribute in DataFrame: {self.sa_attribute}")

        if self._auto_detect_types_needed:
            self._numerical_qi = []
            self._categorical_qi = []
            for col in self.qi_attributes:
                 # Simple check: if dtype is numeric (int, float) treat as numerical
                 # Otherwise, treat as categorical. Could be refined.
                if pd.api.types.is_numeric_dtype(df_copy[col]):
                    self._numerical_qi.append(col)
                    # Ensure floats for consistent range calculations if integers are present
                    if pd.api.types.is_integer_dtype(df_copy[col]):
                         df_copy[col] = df_copy[col].astype(float)
                else:
                    self._categorical_qi.append(col)
            print(f"Auto-detected Numerical QIs: {self._numerical_qi}")
            print(f"Auto-detected Categorical QIs: {self._categorical_qi}")
            self._auto_detect_types_needed = False # Mark as detected

        # Ensure correct types based on specification
        for col in self._numerical_qi:
            if not pd.api.types.is_numeric_dtype(df_copy[col]):
                 try:
                     df_copy[col] = pd.to_numeric(df_copy[col])
                     print(f"Warning: Column '{col}' specified as numerical but wasn't; converted.")
                 except ValueError:
                      raise ValueError(f"Column '{col}' specified as numerical but could not be converted.")
            # Convert integers to float for range calculation consistency
            if pd.api.types.is_integer_dtype(df_copy[col]):
                 df_copy[col] = df_copy[col].astype(float)
        for col in self._categorical_qi:
            if not isinstance(df_copy[col], pd.CategoricalDtype) and not pd.api.types.is_object_dtype(df_copy[col]) and not pd.api.types.is_string_dtype(df_copy[col]):
                print(f"Warning: Column '{col}' specified as categorical might have an unexpected type ({df_copy[col].dtype}). Converting to string.")
            df_copy[col] = df_copy[col].astype(str) # Ensure strings for consistent generalization

        return df_copy


    def _choose_split_attribute(self, df_partition: pd.DataFrame) -> Optional[str]:
        """Chooses the best QI attribute to split on (widest normalized range/most categories)."""
        best_attr = None
        max_spread = -1.0

        for attr in self._numerical_qi:
            if df_partition[attr].nunique() > 1:
                min_val, max_val = df_partition[attr].min(), df_partition[attr].max()
                # Normalize range by overall range in the original data (approximation)
                if self._original_df_for_loss is None:
                    raise ValueError("Original DataFrame not provided. Cannot normalize range.")
                
                global_min = self._original_df_for_loss[attr].min() # Requires storing original df
                global_max = self._original_df_for_loss[attr].max()
                global_range = global_max - global_min
                normalized_range = (max_val - min_val) / global_range if global_range > 0 else 0
                if normalized_range > max_spread:
                    max_spread = normalized_range
                    best_attr = attr

        for attr in self._categorical_qi:
            n_unique = df_partition[attr].nunique()
            if n_unique > 1:
                 # Prioritize splitting categoricals with more distinct values
                 # Normalize by total categories (approximation)
                if self._original_df_for_loss is None:
                    raise ValueError("Original DataFrame not provided. Cannot normalize range.")
                global_unique = self._original_df_for_loss[attr].nunique()
                normalized_uniqueness = n_unique / global_unique if global_unique > 0 else 0
                # Give slight preference to splitting categoricals if range is similar
                if normalized_uniqueness >= max_spread * 0.9: # Heuristic threshold
                    max_spread = normalized_uniqueness
                    best_attr = attr

        return best_attr

    def _find_split_value(self, df_partition: pd.DataFrame, attr: str) -> Any:
        """Find the median for numerical or a category split point."""
        if attr in self._numerical_qi:
            return df_partition[attr].median()
        elif attr in self._categorical_qi:
            # Simple split: take the first category (can be improved)
            unique_sorted = sorted(df_partition[attr].unique())
            # Split roughly in the middle category value if possible
            return unique_sorted[len(unique_sorted) // 2]
        else:
            raise TypeError(f"Attribute {attr} is not defined as numerical or categorical.")


    def _recursive_partition(self, df_indices: pd.Index, level: int):
        """Recursively splits partitions until k-anonymity or max level is reached."""
        if level >= self.max_split_level or len(df_indices) < 2 * self.k:
            # Stop condition: max depth reached or partition too small to split further
            # Or, check if all QI values are already identical (no more splits possible)
             all_identical = True
             partition_data = self._working_df.loc[df_indices, self.qi_attributes]
             if len(partition_data) > 0 : # Check if df is not empty
                 for col in self.qi_attributes:
                     if partition_data[col].nunique() > 1:
                         all_identical = False
                         break
             if all_identical and len(df_indices)>0 : # Also stop if no more differentiating QIs
                 #print(f"Stopping partition size {len(df_indices)} at level {level} - all QIs identical.") # Debug
                 pass # Fall through to add partition below
             elif len(df_indices) < self.k:
                  print(f"WARNING: Partition with {len(df_indices)} records (less than k={self.k}) could not be merged/split further at level {level}. May violate k-anonymity.")
                  # In a more robust implementation, try merging this with a sibling or parent,
                  # or apply stronger suppression. Here, we issue a warning and keep it.

             self.partitions.append(df_indices)
             #print(f"Final Partition (Size: {len(df_indices)}, Level: {level}) Indices: {df_indices.tolist()}") # Debug
             return

        # Choose attribute and value to split on
        current_partition_df = self._working_df.loc[df_indices]
        split_attr = self._choose_split_attribute(current_partition_df)

        if split_attr is None:
            # No attribute found to split on (all values might be identical)
            self.partitions.append(df_indices)
            #print(f"Final Partition (Size: {len(df_indices)}, Level: {level}) - No split attr. Indices: {df_indices.tolist()}") # Debug
            return

        split_value = self._find_split_value(current_partition_df, split_attr)

        # Perform the split
        if split_attr in self._numerical_qi:
            left_indices = current_partition_df[current_partition_df[split_attr] <= split_value].index
            right_indices = current_partition_df[current_partition_df[split_attr] > split_value].index
        else: # Categorical split (split based on the chosen category value)
            # Simple split: <= split_value goes left, > goes right (alphabetically)
            # Can refine this based on hierarchy or frequency
            left_indices = current_partition_df[current_partition_df[split_attr] <= split_value].index
            right_indices = current_partition_df[current_partition_df[split_attr] > split_value].index

        # Ensure both resulting partitions meet k or cannot be split further reasonably
        valid_split = True
        if len(left_indices) < self.k or len(right_indices) < self.k:
             # If a split violates k, stop partitioning here
             valid_split = False
             # Could try other split attributes/values here for more robustness

        if valid_split:
            #print(f"Level {level}: Splitting partition size {len(df_indices)} on '{split_attr}' ({len(left_indices)} / {len(right_indices)})") # Debug
            self._recursive_partition(left_indices, level + 1)
            self._recursive_partition(right_indices, level + 1)
        else:
            # Stop splitting if it violates k-anonymity
            self.partitions.append(df_indices)
            #print(f"Final Partition (Size: {len(df_indices)}, Level: {level}) - Split violates k. Indices: {df_indices.tolist()}") # Debug


    def fit(self, df: pd.DataFrame):
        """Fit the model to the DataFrame, partitioning it according to k-anonymity."""
        self._original_df_for_loss = df.copy() # Store for loss calculation
        self._working_df = self._validate_and_prepare_df(df)

        # Reset state
        self.partitions = []
        self.generalization_map = []

        print("Starting recursive partitioning...")
        self._recursive_partition(self._working_df.index, level=0)
        print(f"Partitioning complete. Found {len(self.partitions)} partitions.")

        # Calculate generalization for each partition
        print("Calculating generalizations for partitions...")
        for indices in self.partitions:
            partition_data = self._working_df.loc[indices]
            gen_dict = {}
            if not partition_data.empty:
                 for attr in self.qi_attributes:
                     if attr in self._numerical_qi:
                         gen_dict[attr] = generalize_numeric(partition_data[attr])
                     elif attr in self._categorical_qi:
                         # Pass hierarchy if available in future
                         gen_dict[attr] = generalize_categorical(partition_data[attr])
            else: # Handle empty partitions if they occur (shouldn't ideally)
                 for attr in self.qi_attributes:
                      gen_dict[attr] = '*' # Or pd.NA
            self.generalization_map.append(gen_dict)
        print("Generalization calculation complete.")
        return self


    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the transformation to the DataFrame to achieve k-anonymity."""
        if not self.partitions:
            raise RuntimeError("Fit method must be called before transform.")

        # Create a copy to store the anonymized results
        self.anonymized_data = df.copy()

        print("Applying generalizations to data...")
        # Apply generalization based on partition map
        if len(self.partitions) != len(self.generalization_map):
             raise RuntimeError("Mismatch between partitions and generalization map. Refit needed.")

        for i, indices in enumerate(self.partitions):
            if not indices.empty: # Check if indices list is not empty
                generalizations = self.generalization_map[i]
                for attr, gen_value in generalizations.items():
                    # Use .loc to modify the DataFrame slice
                    self.anonymized_data[attr] = self.anonymized_data[attr].astype(str)
                    self.anonymized_data.loc[indices, attr] = gen_value
            #else: print(f"Skipping empty partition index {i}") # Debug

        print("Anonymization transformation complete.")
        return self.anonymized_data

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fits the model and then transforms the data."""
        self.fit(df)
        return self.transform(df)

    def get_information_loss(self) -> Optional[float]:
        """Calculate the information loss after transformation."""
        if self.anonymized_data is None or self._original_df_for_loss is None:
             print("Warning: Cannot calculate information loss. Call fit() and transform() first.")
             return None
        if len(self._original_df_for_loss) != len(self.anonymized_data):
             print("Warning: Original and anonymized data length mismatch. Cannot calculate loss accurately.")
             return None

        return calculate_information_loss(
             self._original_df_for_loss,
             self.anonymized_data,
             self.qi_attributes,
             self._numerical_qi,
             self._categorical_qi
        )
