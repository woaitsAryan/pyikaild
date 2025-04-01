"""
This module implements the Improved l-Diversity (ILD) algorithm.

The ILD algorithm ensures that each equivalence class has at least 'l' distinct values for the Sensitive Attribute.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Any

# ---------------------
# ILD Class
# ---------------------
class ILD:
    """
    Implement Improved l-Diversity (ILD).

    Ensure that each equivalence class (group of records with identical
    generalized Quasi-Identifiers) has at least 'l' distinct values for the
    Sensitive Attribute (SA).

    This class should be applied *after* k-anonymization. It modifies the
    sensitive attribute in groups violating l-diversity by borrowing values
    from other groups, as described in the paper.

    **Warning:** Modifying sensitive attributes can impact data utility and
    integrity. Use with caution and understand the implications.

    Attributes:
        l (int): The minimum number of distinct sensitive values required per group.
        qi_attributes (List[str]): List of QI column names (should match those used in IKA).
        sa_attribute (str): Column name of the Sensitive Attribute.
    """

    def __init__(self, l: int, qi_attributes: List[str], sa_attribute: str, silent: bool = False):
        """Initialize the ILD class with the given parameters."""
        if l < 2:
            # l=1 diversity is meaningless (always satisfied if SA exists)
            raise ValueError("l must be at least 2 for l-diversity to be meaningful.")
        if not qi_attributes:
            raise ValueError("qi_attributes list cannot be empty.")
        if not sa_attribute:
            raise ValueError("sa_attribute must be specified.")

        self.l = l
        self.qi_attributes = qi_attributes
        self.sa_attribute = sa_attribute
        self._diverse_data: Optional[pd.DataFrame] = None
        self.silent = silent
        
    def _print(self, *args, **kwargs):
        """Prints only if not silent."""
        if not self.silent:
            print(*args, **kwargs)

    def _validate_df(self, df: pd.DataFrame):
        """Check if required columns exist."""
        missing_qi = [col for col in self.qi_attributes if col not in df.columns]
        if missing_qi:
            raise ValueError(f"Missing QI attributes in DataFrame: {missing_qi}")
        if self.sa_attribute not in df.columns:
            raise ValueError(f"Missing SA attribute in DataFrame: {self.sa_attribute}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply l-diversity enforcement to a k-anonymized DataFrame."""
        self._validate_df(df)
        self._diverse_data = df.copy()

        self._print(f"Applying l-diversity (l={self.l})...")
        grouped = self._diverse_data.groupby(self.qi_attributes, observed=False) # observed=False is safer with categoricals
        all_groups = list(grouped.groups.keys()) # Get all group identifiers
        num_groups = len(all_groups)
        
        # Use numerical indices for easier random selection later
        group_indices_map = {i: name for i, name in enumerate(all_groups)} 
        # Reverse map for finding index from name if needed
        group_name_map = {name: i for i, name in group_indices_map.items()} 

        violating_group_names = []
        valid_groups_info = {} # Store SA values of valid groups

        self._print("Identifying groups violating l-diversity...")
        for name, group in grouped:
            unique_sa_count = group[self.sa_attribute].nunique()
            if unique_sa_count < self.l:
                violating_group_names.append(name)
            else:
                valid_groups_info[name] = set(group[self.sa_attribute].unique())

        self._print(f"Found {len(violating_group_names)} groups violating l={self.l} diversity.")

        if not violating_group_names:
            self._print("No l-diversity violations found.")
            return self._diverse_data

        if not valid_groups_info:
             self._print(f"WARNING: No valid l-diverse groups found to borrow SA values from. Cannot enforce l={self.l}. Returning original data.")
             return self._diverse_data # Cannot proceed

        self._print("Attempting to fix violating groups...")
        processed_violations = 0
        for group_name in violating_group_names:
            group_df = grouped.get_group(group_name)
            group_indices = group_df.index
            current_sa_values = set(group_df[self.sa_attribute].unique())
            needed_distinct_values = self.l - len(current_sa_values)

            potential_donors = list(valid_groups_info.keys())
            if not potential_donors:
                 print(f"Warning: No more valid donor groups available for group {group_name}. Skipping.")
                 continue # No donors left

            # Find SA values to borrow
            borrowed_values: set[str] = set()
            donor_groups_used = set()

            attempts = 0
            max_attempts = num_groups * 2 # Limit attempts to find donors

            while len(borrowed_values) < needed_distinct_values and attempts < max_attempts:
                attempts += 1
                # Choose a random valid donor group
                idx = np.random.randint(len(potential_donors))
                donor_group_name = potential_donors[idx]

                # Avoid using same donor repeatedly unless necessary
                if donor_group_name in donor_groups_used and len(potential_donors) > 1:
                    # Try a different donor if available
                    if attempts % 5 == 0: # Periodically allow reusing donor if stuck
                         pass
                    else:
                        continue

                donor_sa_values = valid_groups_info[donor_group_name]
                # Find a value in donor that's not already in the target group or borrowed
                candidate_values = donor_sa_values - current_sa_values - borrowed_values

                if candidate_values:
                    # Borrow one value
                    value_to_borrow = np.random.choice(list(candidate_values))
                    borrowed_values.add(value_to_borrow)
                    donor_groups_used.add(donor_group_name)
                    #print(f"  Borrowing '{value_to_borrow}' from group {donor_group_name} for group {group_name}") # Debug
                #else: print(f"  Donor {donor_group_name} had no new SA values.") #Debug

                # If a donor group's values are exhausted for this target, remove it from potential donors list temporarily
                if not (donor_sa_values - current_sa_values - borrowed_values) and len(potential_donors)>1:
                     potential_donors.pop(idx) # Remove exhausted donor if others exist

            if len(borrowed_values) < needed_distinct_values:
                self._print(f"Warning: Could not find enough distinct SA values ({len(borrowed_values)} found, {needed_distinct_values} needed) to borrow for group {group_name}. L-diversity might not be fully enforced.")
                # Optionally: Apply suppression '*' to some records instead?

            # Modify records in the violating group
            records_to_modify_indices = np.random.choice(group_indices, size=len(borrowed_values), replace=False)

            for i, record_index in enumerate(records_to_modify_indices):
                value_to_assign = list(borrowed_values)[i]
                #print(f"  Modifying record index {record_index} SA to '{value_to_assign}'") # Debug
                self._diverse_data.loc[record_index, self.sa_attribute] = value_to_assign
            
            processed_violations += 1

        self._print(f"Processed {processed_violations} violating groups.")
        # Verify l-diversity after modification (optional)
        # final_grouped = self._diverse_data.groupby(self.qi_attributes)
        # final_violations = 0
        # for name, group in final_grouped:
        #      if group[self.sa_attribute].nunique() < self.l:
        #           final_violations += 1
        # print(f"Verification: {final_violations} groups still violate l-diversity.")

        return self._diverse_data

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the data to enforce l-diversity. Fit is not needed."""
        # No fitting step required for ILD as defined here
        return self.transform(df)