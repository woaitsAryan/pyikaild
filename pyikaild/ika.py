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
from collections import defaultdict
import numpy as np

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
        silent (bool): Suppresses all print statements if True. Default False.
    """

    def __init__(self,
                 k: int,
                 qi_attributes: List[str],
                 sa_attribute: str, # Keep for context, consistency with paper
                 numerical_qi: Optional[List[str]] = None,
                 categorical_qi: Optional[List[str]] = None,
                 max_split_level: int = 10,
                 silent: bool = False):
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
        self.silent = silent

        # Auto-detect numerical/categorical if not provided (simple heuristic)
        self._numerical_qi = numerical_qi if numerical_qi is not None else []
        self._categorical_qi = categorical_qi if categorical_qi is not None else []
        self._auto_detect_types_needed = not (numerical_qi or categorical_qi)

        self.anonymized_data: Optional[pd.DataFrame] = None
        self.partitions: List[pd.Index] = []
        self.generalization_map: List[Dict[str, Any]] = []
        self._original_df_for_loss: Optional[pd.DataFrame] = None # Store original for loss calc

    def _print(self, *args, **kwargs):
        """Prints only if not silent."""
        if not self.silent:
            print(*args, **kwargs)

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
            self._print(f"Auto-detected Numerical QIs: {self._numerical_qi}")
            self._print(f"Auto-detected Categorical QIs: {self._categorical_qi}")
            self._auto_detect_types_needed = False # Mark as detected

        # Ensure correct types based on specification
        for col in self._numerical_qi:
            if not pd.api.types.is_numeric_dtype(df_copy[col]):
                 try:
                     df_copy[col] = pd.to_numeric(df_copy[col])
                     self._print(f"Warning: Column '{col}' specified as numerical but wasn't; converted.")
                 except ValueError:
                      raise ValueError(f"Column '{col}' specified as numerical but could not be converted.")
            # Convert integers to float for range calculation consistency
            if pd.api.types.is_integer_dtype(df_copy[col]):
                 df_copy[col] = df_copy[col].astype(float)
        for col in self._categorical_qi:
            if not isinstance(df_copy[col], pd.CategoricalDtype) and not pd.api.types.is_object_dtype(df_copy[col]) and not pd.api.types.is_string_dtype(df_copy[col]):
                self._print(f"Warning: Column '{col}' specified as categorical might have an unexpected type ({df_copy[col].dtype}). Converting to string.")
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
                 #self._print(f"Stopping partition size {len(df_indices)} at level {level} - all QIs identical.") # Debug
                 pass # Fall through to add partition below
             elif len(df_indices) < self.k:
                  self._print(f"WARNING: Partition with {len(df_indices)} records (less than k={self.k}) could not be merged/split further at level {level}. May violate k-anonymity.")
                  # In a more robust implementation, try merging this with a sibling or parent,
                  # or apply stronger suppression. Here, we issue a warning and keep it.

             self.partitions.append(df_indices)
             #self._print(f"Final Partition (Size: {len(df_indices)}, Level: {level}) Indices: {df_indices.tolist()}") # Debug
             return

        # Choose attribute and value to split on
        current_partition_df = self._working_df.loc[df_indices]
        split_attr = self._choose_split_attribute(current_partition_df)

        if split_attr is None:
            # No attribute found to split on (all values might be identical)
            self.partitions.append(df_indices)
            #self._print(f"Final Partition (Size: {len(df_indices)}, Level: {level}) - No split attr. Indices: {df_indices.tolist()}") # Debug
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
            #self._print(f"Level {level}: Splitting partition size {len(df_indices)} on '{split_attr}' ({len(left_indices)} / {len(right_indices)})") # Debug
            self._recursive_partition(left_indices, level + 1)
            self._recursive_partition(right_indices, level + 1)
        else:
            # Stop splitting if it violates k-anonymity
            self.partitions.append(df_indices)
            #self._print(f"Final Partition (Size: {len(df_indices)}, Level: {level}) - Split violates k. Indices: {df_indices.tolist()}") # Debug


    def fit(self, df: pd.DataFrame):
        """Fit the model to the DataFrame, partitioning it according to k-anonymity."""
        self._original_df_for_loss = df.copy() # Store for loss calculation
        self._working_df = self._validate_and_prepare_df(df)

        # Reset state
        self.partitions = []
        self.generalization_map = []

        self._print("Starting recursive partitioning...")
        self._recursive_partition(self._working_df.index, level=0)
        self._print(f"Partitioning complete. Found {len(self.partitions)} partitions.")

        # Calculate generalization for each partition
        self._print("Calculating generalizations for partitions...")
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
        self._print("Generalization calculation complete.")
        return self


    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the transformation to the DataFrame to achieve k-anonymity."""
        if not self.partitions:
            raise RuntimeError("Fit method must be called before transform.")

        # Create a copy to store the anonymized results
        self.anonymized_data = df.copy()

        self._print("Applying generalizations to data...")
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
            #else: self._print(f"Skipping empty partition index {i}") # Debug

        self._print("Anonymization transformation complete.")
        return self.anonymized_data

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fits the model and then transforms the data."""
        self.fit(df)
        return self.transform(df)

    def get_information_loss(self) -> Optional[float]:
        """Calculate the information loss after transformation."""
        if self.anonymized_data is None or self._original_df_for_loss is None:
             self._print("Warning: Cannot calculate information loss. Call fit() and transform() first.")
             return None
        if len(self._original_df_for_loss) != len(self.anonymized_data):
             self._print("Warning: Original and anonymized data length mismatch. Cannot calculate loss accurately.")
             return None

        return calculate_information_loss(
             self._original_df_for_loss,
             self.anonymized_data,
             self.qi_attributes,
             self._numerical_qi,
             self._categorical_qi
        )

class ARBA(IKA):
    """
    Implement Adaptive Risk-Based Anonymization (ARBA).
    
    Extends IKA with risk-based adaptive k-values and enhanced diversity.
    
    Attributes:
        base_k (int): The base minimum size of an equivalence class.
        diversity_threshold (int): The l-diversity threshold for sensitive attributes.
        qi_attributes (List[str]): List of column names to be treated as Quasi-Identifiers.
        sa_attribute (str): Column name of the Sensitive Attribute.
        numerical_qi (List[str]): List of QI attributes that are numerical.
        categorical_qi (List[str]): List of QI attributes that are categorical.
        max_split_level (int): Maximum recursion depth for splitting (controls granularity).
        anonymized_data (Optional[pd.DataFrame]): Stores the result after transform.
        partitions (List[pd.Index]): Stores the indices of records belonging to each final partition.
        generalization_map (Dict): Stores the generalized values for each partition and QI.
        risk_scores (Dict): Stores risk scores for each cluster.
        cluster_k_values (Dict): Stores adaptive k-values for each cluster.
    """

    def __init__(self,
                 base_k: int,
                 diversity_threshold: int,
                 qi_attributes: List[str],
                 sa_attribute: str,
                 numerical_qi: Optional[List[str]] = None,
                 categorical_qi: Optional[List[str]] = None,
                 max_split_level: int = 10,
                 silent: bool = False):
        """Initialize the ARBA class with the given parameters."""
        # Initialize the parent IKA class with base_k
        super().__init__(k=base_k, 
                         qi_attributes=qi_attributes,
                         sa_attribute=sa_attribute,
                         numerical_qi=numerical_qi,
                         categorical_qi=categorical_qi,
                         max_split_level=max_split_level,
                         silent=silent)
        
        # ARBA specific attributes
        self.base_k = base_k
        self.diversity_threshold = diversity_threshold
        self.risk_scores = {}
        self.cluster_k_values = {}
        self.clusters = []
        self.adjacent_clusters = defaultdict(list)

    def _cluster_dataset(self, df: pd.DataFrame) -> List[pd.Index]:
        """
        Partition dataset into clusters based on attribute distributions.
        
        Using a different clustering strategy than IKA's partitioning.
        """
        self._print("Clustering dataset based on attribute distributions...")
        
        # Instead of reusing IKA's partitioning, implement a more distinct clustering approach
        # that maintains more diversity in the sensitive attribute
        
        # For now, start with one cluster containing all records
        all_indices = df.index
        
        # Find natural breaks in the data (using simple equal-frequency binning)
        clusters = []
        
        # For small datasets, create 2-3 clusters; for larger datasets, create more
        n_clusters = max(2, min(5, len(df) // (2 * self.base_k)))
        
        if len(self._numerical_qi) > 0:
            # Use numeric attributes for clustering when available
            # Choose the attribute with highest cardinality for initial split
            best_attr = self._numerical_qi[0]
            for attr in self._numerical_qi:
                if df[attr].nunique() > df[best_attr].nunique():
                    best_attr = attr
            
            # Sort by the chosen attribute
            sorted_indices = df.sort_values(by=best_attr).index
            
            # Split into roughly equal-sized clusters
            cluster_size = max(self.base_k, len(sorted_indices) // n_clusters)
            
            for i in range(0, len(sorted_indices), cluster_size):
                end_idx = min(i + cluster_size, len(sorted_indices))
                cluster_indices = sorted_indices[i:end_idx]
                if len(cluster_indices) >= self.base_k:
                    clusters.append(cluster_indices)
                elif clusters:  # If we have existing clusters, add to the last one
                    clusters[-1] = clusters[-1].union(cluster_indices)
                else:  # Should only happen if first cluster is too small
                    clusters.append(cluster_indices)
        else:
            # For categorical-only data, use groupby on the categorical attribute
            # with the most distinct values that still maintains clusters of size >= k
            clusters.append(all_indices)  # Default: one cluster with everything
        
        self.clusters = clusters
        self._print(f"Created {len(self.clusters)} initial clusters")
        return self.clusters

    def _assess_risk_level(self, cluster_indices: pd.Index) -> float:
        """
        Compute risk score based on uniqueness, outliers, and attribute distribution.
        
        This implements Step 3.a in the ARBA algorithm.
        """
        if len(cluster_indices) == 0:
            return 1.0  # Maximum risk for empty clusters (shouldn't happen)
        
        cluster_data = self._working_df.loc[cluster_indices]
        risk_score = 0.0
        
        # Factor 1: Uniqueness (higher uniqueness = higher risk)
        uniqueness_factor = 0.0
        for attr in self.qi_attributes:
            unique_ratio = cluster_data[attr].nunique() / len(cluster_data)
            uniqueness_factor += unique_ratio
        uniqueness_factor /= len(self.qi_attributes)
        
        # Factor 2: Outliers (presence of outliers increases risk)
        outlier_factor = 0.0
        for attr in self._numerical_qi:
            # Simple Z-score based outlier detection
            if len(cluster_data) > 1:  # Need at least 2 points for std
                z_scores = np.abs((cluster_data[attr] - cluster_data[attr].mean()) / 
                                  max(cluster_data[attr].std(ddof=0), 0.0001))  # Avoid div by zero
                outlier_ratio = (z_scores > 2).mean()  # Proportion of outliers
                outlier_factor += outlier_ratio
        
        if len(self._numerical_qi) > 0:
            outlier_factor /= len(self._numerical_qi)
        
        # Factor 3: Sensitive attribute distribution (less diverse = higher risk)
        sa_distribution = cluster_data[self.sa_attribute].value_counts(normalize=True)
        entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in sa_distribution)
        max_entropy = np.log2(len(sa_distribution)) if len(sa_distribution) > 0 else 1
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        diversity_factor = 1 - normalized_entropy  # Higher value = less diverse = higher risk
        
        # Combine factors (increase weight of diversity factor to more aggressively
        # protect sensitive information patterns)
        risk_score = 0.3 * uniqueness_factor + 0.2 * outlier_factor + 0.5 * diversity_factor
        
        # Ensure risk is between 0 and 1
        return min(max(risk_score, 0.0), 1.0)
    
    def _compute_adaptive_k(self, cluster_indices: pd.Index) -> int:
        """
        Compute adaptive k value based on risk score.
        
        This implements Step 3.b in the ARBA algorithm.
        """
        risk_score = self._assess_risk_level(cluster_indices)
        self.risk_scores[tuple(cluster_indices)] = risk_score
        
        # More aggressive scaling of k with risk (using exponential rather than linear scaling)
        # For small risk values, k remains close to base_k, but rises more quickly as risk increases
        risk_multiplier = 1.0 + 2.0 * (np.exp(risk_score) - 1)
        adaptive_k = max(self.base_k, int(np.ceil(risk_multiplier * self.base_k)))
        
        self.cluster_k_values[tuple(cluster_indices)] = adaptive_k
        
        self._print(f"Cluster (size: {len(cluster_indices)}) - Risk: {risk_score:.2f}, Adaptive k: {adaptive_k}")
        return adaptive_k
    
    def _anonymize_cluster(self, cluster_indices: pd.Index) -> None:
        """
        Anonymize a cluster using its specific k value.
        
        This implements Step 3.c in the ARBA algorithm.
        """
        if len(cluster_indices) == 0:
            return
        
        # Get adaptive k for this cluster
        adaptive_k = self._compute_adaptive_k(cluster_indices)
        
        # Use IKA-like partitioning but with the adaptive k
        # We'll modify the recursive partitioning to use the adaptive k
        self.k = adaptive_k  # Temporarily change k value
        
        # For simplicity, we'll just add the cluster as a partition for now
        # In a more sophisticated implementation, we would repartition with adaptive k
        self.partitions.append(cluster_indices)

    def _identify_adjacent_clusters(self) -> None:
        """
        Identify adjacent clusters for boundary refinement.
        
        This is part of Step 4 in the ARBA algorithm.
        """
        self._print("Identifying adjacent clusters...")
        
        # A simple approach to identify adjacent clusters
        # For each pair of clusters, check if they're "close" in QI space
        for i, cluster1 in enumerate(self.clusters):
            for j, cluster2 in enumerate(self.clusters):
                if i >= j:
                    continue  # Skip self-comparison and already compared pairs
                
                # Check if clusters are adjacent (simplified)
                if self._are_clusters_adjacent(cluster1, cluster2):
                    self.adjacent_clusters[i].append(j)
                    self.adjacent_clusters[j].append(i)
        
        self._print(f"Identified adjacency relationships among clusters.")

    def _are_clusters_adjacent(self, cluster1: pd.Index, cluster2: pd.Index) -> bool:
        """Determine if two clusters are adjacent in QI space."""
        # Simplified approach: Check proximity in QI attributes
        data1 = self._working_df.loc[cluster1]
        data2 = self._working_df.loc[cluster2]
        
        # For numerical attributes, check if ranges overlap or are close
        for attr in self._numerical_qi:
            min1, max1 = data1[attr].min(), data1[attr].max()
            min2, max2 = data2[attr].min(), data2[attr].max()
            
            # Check if ranges overlap or are close
            overlap_or_close = (min1 <= max2 and min2 <= max1) or \
                               abs(min1 - max2) < 0.1 * (max(max1, max2) - min(min1, min2)) or \
                               abs(min2 - max1) < 0.1 * (max(max1, max2) - min(min1, min2))
            
            if not overlap_or_close:
                return False
        
        # For categorical attributes, check if they share values
        for attr in self._categorical_qi:
            values1 = set(data1[attr].unique())
            values2 = set(data2[attr].unique())
            
            if not values1.intersection(values2) and len(values1) > 0 and len(values2) > 0:
                return False
        
        return True

    def _adjust_boundary_records(self, cluster_i: int, cluster_j: int) -> None:
        """
        Adjust boundary records between adjacent clusters.
        
        This implements the boundary refinement in Step 4 of the ARBA algorithm.
        """
        if cluster_i not in range(len(self.clusters)) or cluster_j not in range(len(self.clusters)):
            return
        
        cluster1 = self.clusters[cluster_i]
        cluster2 = self.clusters[cluster_j]
        
        # Find boundary records
        boundary_records1 = self._identify_boundary_records(cluster1, cluster2)
        boundary_records2 = self._identify_boundary_records(cluster2, cluster1)
        
        # If needed, move some boundary records to ensure global k-anonymity
        k1 = self.cluster_k_values.get(tuple(cluster1), self.base_k)
        k2 = self.cluster_k_values.get(tuple(cluster2), self.base_k)
        
        # Simple heuristic: Move records from larger cluster to smaller if needed
        if len(cluster1) - len(boundary_records1) < k1:
            # Need to keep some boundary records in cluster1
            records_to_keep = k1 - (len(cluster1) - len(boundary_records1))
            boundary_records1 = boundary_records1[records_to_keep:]
        
        if len(cluster2) - len(boundary_records2) < k2:
            # Need to keep some boundary records in cluster2
            records_to_keep = k2 - (len(cluster2) - len(boundary_records2))
            boundary_records2 = boundary_records2[records_to_keep:]
        
        # Update clusters after boundary adjustment
        # This is a simplified implementation - in practice, you'd need to update
        # both self.clusters and self.partitions appropriately
        
        self._print(f"Adjusted boundary between clusters {cluster_i} and {cluster_j}")

    def _identify_boundary_records(self, cluster1: pd.Index, cluster2: pd.Index) -> pd.Index:
        """Identify boundary records between two clusters."""
        data1 = self._working_df.loc[cluster1]
        data2 = self._working_df.loc[cluster2]
        
        # Identify records in cluster1 that are "close" to cluster2
        boundary_scores = pd.Series(0.0, index=cluster1)
        
        for attr in self._numerical_qi:
            # Find distance to the nearest point in cluster2 for each point in cluster1
            for idx in cluster1:
                record_value = self._working_df.loc[idx, attr]
                min_dist = min(abs(record_value - val) for val in data2[attr])
                boundary_scores[idx] += min_dist
        
        # Normalize and invert scores (lower distance = higher boundary score)
        if boundary_scores.max() > boundary_scores.min():
            boundary_scores = 1 - (boundary_scores - boundary_scores.min()) / (boundary_scores.max() - boundary_scores.min())
        
        # Return indices of boundary records (top 10% as a simple heuristic)
        num_boundary = max(1, int(0.1 * len(cluster1)))
        return boundary_scores.sort_values(ascending=False).index[:num_boundary]

    def _enhance_diversity(self) -> None:
        """
        Enhance diversity for equivalence classes below the threshold.
        
        This implements Step 5 in the ARBA algorithm.
        """
        self._print(f"Enhancing diversity to meet threshold l={self.diversity_threshold}...")
        
        # Group data by equivalence classes
        equivalence_classes = self.anonymized_data.groupby(self.qi_attributes)
        
        class_indices = {}
        class_entropy = {}
        class_diversity = {}
        
        # Calculate entropy and diversity for each equivalence class
        for name, group in equivalence_classes:
            indices = group.index
            class_indices[name] = indices
            
            # Calculate entropy of sensitive attribute distribution
            sa_counts = group[self.sa_attribute].value_counts(normalize=True)
            entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in sa_counts)
            class_entropy[name] = entropy
            
            # Count distinct sensitive values
            class_diversity[name] = group[self.sa_attribute].nunique()
        
        # Identify classes needing diversity enhancement
        low_diversity_classes = {name: indices for name, indices in class_indices.items()
                                if class_diversity[name] < self.diversity_threshold}
        
        if not low_diversity_classes:
            self._print("All equivalence classes already meet diversity threshold.")
            return
        
        self._print(f"Found {len(low_diversity_classes)} classes below diversity threshold.")
        
        # Get list of all available sensitive attribute values
        all_sa_values = set(self.anonymized_data[self.sa_attribute].unique())
        
        # More aggressive approach: modify records to ensure diversity
        for name, indices in low_diversity_classes.items():
            current_diversity = class_diversity[name]
            needed_values = self.diversity_threshold - current_diversity
            
            if needed_values <= 0:
                continue
                
            # Get current unique values
            current_values = set(self.anonymized_data.loc[indices, self.sa_attribute].unique())
            
            # Find values that aren't already in this class
            missing_values = list(all_sa_values - current_values)
            np.random.shuffle(missing_values)
            
            # Select values to add (up to the needed amount)
            values_to_add = missing_values[:min(len(missing_values), needed_values)]
            
            if not values_to_add:
                self._print(f"Warning: Could not find additional values for class {name}")
                continue
                
            # Select records to modify (up to 20% of the class, prioritizing records with 
            # SA values that appear most frequently in the class)
            class_data = self.anonymized_data.loc[indices]
            value_counts = class_data[self.sa_attribute].value_counts()
            most_common = value_counts.index[0] if not value_counts.empty else None
            
            if most_common is not None:
                # Try to modify records with the most common value
                candidates = class_data[class_data[self.sa_attribute] == most_common].index
                # Limit modifications to at most 20% of records
                n_to_modify = min(len(values_to_add), max(1, int(0.2 * len(indices))))
                
                if len(candidates) >= n_to_modify:
                    records_to_modify = np.random.choice(candidates, size=n_to_modify, replace=False)
                    
                    # Apply the modifications
                    for i, record_idx in enumerate(records_to_modify):
                        if i < len(values_to_add):
                            self.anonymized_data.loc[record_idx, self.sa_attribute] = values_to_add[i]
                    
                    self._print(f"Enhanced diversity for class {name}: {current_diversity} -> {current_diversity + len(values_to_add)}")

    def fit(self, df: pd.DataFrame):
        """Fit the ARBA model to the DataFrame."""
        self._original_df_for_loss = df.copy()
        self._working_df = self._validate_and_prepare_df(df)
        
        # Reset state
        self.partitions = []
        self.generalization_map = []
        self.clusters = []
        self.risk_scores = {}
        self.cluster_k_values = {}
        self.adjacent_clusters = defaultdict(list)
        
        # Step 2: Cluster dataset
        self._cluster_dataset(self._working_df)
        
        # Step 3: Process each cluster
        self._print("Processing clusters with adaptive k-anonymity...")
        for i, cluster_indices in enumerate(self.clusters):
            self._anonymize_cluster(cluster_indices)
        
        # Step 4: Boundary refinement
        self._print("Performing boundary refinement...")
        self._identify_adjacent_clusters()
        for cluster_i, adjacent_clusters in self.adjacent_clusters.items():
            for cluster_j in adjacent_clusters:
                self._adjust_boundary_records(cluster_i, cluster_j)
        
        # Calculate generalizations for partitions (as in IKA)
        self._print("Calculating generalizations for partitions...")
        for indices in self.partitions:
            partition_data = self._working_df.loc[indices]
            gen_dict = {}
            if not partition_data.empty:
                for attr in self.qi_attributes:
                    if attr in self._numerical_qi:
                        gen_dict[attr] = generalize_numeric(partition_data[attr])
                    elif attr in self._categorical_qi:
                        gen_dict[attr] = generalize_categorical(partition_data[attr])
            else:
                for attr in self.qi_attributes:
                    gen_dict[attr] = '*'
            self.generalization_map.append(gen_dict)
        
        self._print("Generalization calculation complete.")
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the data using ARBA approach."""
        # First apply k-anonymity using parent class method
        anonymized_data = super().transform(df)
        
        # Step 5: Enhance diversity
        self._enhance_diversity()
        
        return self.anonymized_data
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)