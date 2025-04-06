# tests/adult_compare_test.py
from pyikaild.ika import IKA, ARBA
import pandas as pd
import numpy as np
import time
import warnings
import anonypy

def test_adult_comparison():
    print("\n--- Comparing ARBA vs IKA vs anonypy on Adult Dataset ---")
    
    # Load adult data
    adult_colnames = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
        'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
        'hours-per-week', 'native-country', 'income'
    ]
    
    try:
        adult_df_full = pd.read_csv(
            'adult.data',
            header=None,
            names=adult_colnames,
            na_values=' ?',
            skipinitialspace=True
        )
        # Sample for faster testing
        adult_df = adult_df_full.sample(n=1000, random_state=42).dropna()
    except Exception as e:
        print(f"Error loading adult dataset: {e}")
        print("Using synthetic data instead for demonstration")
        # Create synthetic data if adult.data is not available
        np.random.seed(42)
        adult_df = pd.DataFrame({
            'age': np.random.randint(18, 80, 100),
            'workclass': np.random.choice(['Private', 'Self-emp', 'Government', 'Other'], 100),
            'education': np.random.choice(['HS-grad', 'Bachelors', 'Masters', 'Doctorate'], 100),
            'race': np.random.choice(['White', 'Black', 'Asian', 'Other'], 100),
            'sex': np.random.choice(['Male', 'Female'], 100),
            'occupation': np.random.choice(['Professional', 'Clerical', 'Service', 'Manual', 'Sales'], 100)
        })
    
    print(f"\nUsing Adult dataset sample ({len(adult_df)} records)")
    print(adult_df.head())
    
    # Define QIs and SA
    qi_adult = ['age', 'workclass', 'education', 'race', 'sex']
    sa_adult = 'occupation'
    num_qi_adult = ['age']
    cat_qi_adult = ['workclass', 'education', 'race', 'sex']
    
    # 1. Apply IKA with k=5
    print("\n=== Testing IKA (k=5) ===")
    start_time = time.time()
    ika_adult = IKA(k=5, 
                   qi_attributes=qi_adult, 
                   sa_attribute=sa_adult, 
                   numerical_qi=num_qi_adult,
                   categorical_qi=cat_qi_adult,
                   max_split_level=15)  # Limit splits for faster execution
    
    ika_anonymized = ika_adult.fit_transform(adult_df)
    ika_time = time.time() - start_time
    ika_loss = ika_adult.get_information_loss()
    
    print(f"\nIKA Execution Time: {ika_time:.4f} seconds")
    print(f"IKA Information Loss: {ika_loss:.4f}")
    print("\nIKA Anonymized Sample:")
    print(ika_anonymized.head())
    
    # Get diversity info from IKA
    ika_diversity = {}
    for name, group in ika_anonymized.groupby(qi_adult):
        ika_diversity[name] = group[sa_adult].nunique()
    
    print("\nIKA Diversity by group (sample):")
    groups_shown = 0
    for name, diversity in ika_diversity.items():
        if groups_shown < 5:  # Show only 5 groups for brevity
            print(f"Group {name}: {diversity} distinct values")
            groups_shown += 1
    
    # 2. Apply ARBA with base_k=5 and diversity_threshold=3
    print("\n=== Testing ARBA (base_k=5, diversity_threshold=3) ===")
    start_time = time.time()
    arba_adult = ARBA(base_k=5, 
                     diversity_threshold=3,
                     qi_attributes=qi_adult, 
                     sa_attribute=sa_adult, 
                     numerical_qi=num_qi_adult,
                     categorical_qi=cat_qi_adult,
                     max_split_level=15)
    
    arba_anonymized = arba_adult.fit_transform(adult_df)
    arba_time = time.time() - start_time
    arba_loss = arba_adult.get_information_loss()
    
    print(f"\nARBA Execution Time: {arba_time:.4f} seconds")
    print(f"ARBA Information Loss: {arba_loss:.4f}")
    print("\nARBA Anonymized Sample:")
    print(arba_anonymized.head())
    
    # Get diversity info from ARBA
    arba_diversity = {}
    for name, group in arba_anonymized.groupby(qi_adult):
        arba_diversity[name] = group[sa_adult].nunique()
    
    print("\nARBA Diversity by group (sample):")
    groups_shown = 0
    for name, diversity in arba_diversity.items():
        if groups_shown < 5:  # Show only 5 groups for brevity
            print(f"Group {name}: {diversity} distinct values")
            groups_shown += 1
    
    # 3. anonypy comparison
    anonypy_time = None
    anonypy_info_loss = None
    anonypy_diversity = {}
    anonypy_avg_diversity = 0
    
    print("\n=== Testing anonypy (k=5, l=3) ===")
    
    # Create a copy of the dataset for anonypy
    anonypy_df = adult_df.copy()
    
    try:
        # Ensure all numeric columns are properly converted to numeric
        # Force convert all needed columns to appropriate types for anonypy
        for col in num_qi_adult:
            anonypy_df[col] = pd.to_numeric(anonypy_df[col], errors='coerce')
        
        # Handle any NaN values created during conversion
        anonypy_df = anonypy_df.dropna(subset=num_qi_adult)
        
        # Convert all other columns to strings to avoid type issues
        for col in cat_qi_adult:
            anonypy_df[col] = anonypy_df[col].astype("category").cat.codes
        
        # Convert SA to string as well
        anonypy_df[sa_adult] = anonypy_df[sa_adult].astype(str)
        
        # Initialize the Preserver with our dataset
        start_time = time.time()
        
        # Use only a subset of the data for anonypy if needed (for performance)
        # Create a sample for anonypy if the dataset is too large
        if len(anonypy_df) > 500:
            anonypy_sample = anonypy_df.sample(n=500, random_state=42)
        else:
            anonypy_sample = anonypy_df
            
        preserver = anonypy.Preserver(anonypy_sample, qi_adult, sa_adult)
        
        # Get l-diversity (k=5, l=3)
        anonypy_result = preserver.anonymize_l_diversity(k=5, l=3)
        
        anonypy_time = time.time() - start_time
        
        # Create a DataFrame from the result rows
        anonypy_anonymized = pd.DataFrame(anonypy_result)
        
        print(f"\nanonypy Execution Time: {anonypy_time:.4f} seconds")
        
        # Calculate diversity for each generalized group
        unique_qis = set()
        for row in anonypy_result:
            # Convert any list or unhashable values to tuples
            qi_values = []
            for col in qi_adult:
                val = row[col]
                # Convert lists or other unhashable types to string representation
                if isinstance(val, list) or not isinstance(val, (str, int, float, tuple)):
                    val = str(val)
                qi_values.append(val)
            qi_key = tuple(qi_values)
            unique_qis.add(qi_key)
        
        # Calculate diversity for each QI group
        for qi_key in unique_qis:
            sensitive_values = set()
            for row in anonypy_result:
                # Create comparable key
                row_qi_values = []
                for i, col in enumerate(qi_adult):
                    val = row[col]
                    if isinstance(val, list) or not isinstance(val, (str, int, float, tuple)):
                        val = str(val)
                    row_qi_values.append(val)
                
                if tuple(row_qi_values) == qi_key:
                    sensitive_values.add(row[sa_adult])
            anonypy_diversity[qi_key] = len(sensitive_values)
        
        print("\nanonypy Anonymized Result:")
        print(anonypy_anonymized.head())
        
        print("\nanonypy Diversity by group (sample):")
        groups_shown = 0
        for name, diversity in list(anonypy_diversity.items())[:5]:  # Show only 5 groups
            print(f"Group {name}: {diversity} distinct values")
            groups_shown += 1
            
        anonypy_avg_diversity = sum(anonypy_diversity.values()) / len(anonypy_diversity) if anonypy_diversity else 0
        
        # Estimate information loss based on generalization ranges
        # For categorical QIs, we assume 0.5 information loss for simplicity
        info_loss_sum = 0.0
        features_count = 0
        
        for attr in qi_adult:
            if attr in num_qi_adult:
                original_range = float(anonypy_sample[attr].max() - anonypy_sample[attr].min())
                feature_loss = 0.0
                weight = 0
                
                for row in anonypy_result:
                    count = row.get('count', 1)
                    val = row[attr]
                    try:
                        if isinstance(val, str) and '-' in val:
                            min_val, max_val = map(float, val.split('-'))
                            range_size = max_val - min_val
                            feature_loss += (range_size / original_range) * count
                        elif isinstance(val, (int, float)):
                            # Point value has no generalization loss
                            feature_loss += 0
                        else:
                            # Try to convert to float
                            float(val)  # just to test if convertible
                            feature_loss += 0
                    except:
                        feature_loss += 1  # Max loss for invalid values
                    weight += count
                
                if weight > 0:
                    feature_loss /= weight
                    info_loss_sum += feature_loss
                    features_count += 1
            else:
                # For categorical, we'd need to compute semantic distance using hierarchies
                # For simplicity, we'll assume categorical attributes contribute 0.5 to loss
                info_loss_sum += 0.5
                features_count += 1
        
        if features_count > 0:
            anonypy_info_loss = info_loss_sum / features_count
            print(f"anonypy Information Loss (estimated): {anonypy_info_loss:.4f}")
    
    except Exception as e:
        print(f"Error running anonypy: {e}")
        import traceback
        traceback.print_exc()
        # Set a default value to prevent division by zero
        anonypy_info_loss = 1.0
    
    # 4. Compare all results
    print("\n=== Comparison Summary ===")
    print(f"Information Loss: IKA = {ika_loss:.4f}, ARBA = {arba_loss:.4f}" + 
          (f", anonypy ≈ {anonypy_info_loss:.4f}" if anonypy_info_loss is not None else ""))
    
    print(f"Loss Improvement (ARBA vs IKA): {((ika_loss - arba_loss) / ika_loss * 100):.2f}%")
    
    # Fix division by zero by checking if anonypy_info_loss is meaningful
    if anonypy_info_loss is not None and anonypy_info_loss > 0:
        print(f"Loss Improvement (ARBA vs anonypy): {((anonypy_info_loss - arba_loss) / anonypy_info_loss * 100):.2f}%")
    else:
        print("Loss Improvement (ARBA vs anonypy): N/A (anonypy failed)")
    
    print(f"Execution Time: IKA = {ika_time:.4f}s, ARBA = {arba_time:.4f}s" + 
          (f", anonypy = {anonypy_time:.4f}s" if anonypy_time is not None else ""))
    
    # Average diversity
    ika_avg_diversity = sum(ika_diversity.values()) / len(ika_diversity) if ika_diversity else 0
    arba_avg_diversity = sum(arba_diversity.values()) / len(arba_diversity) if arba_diversity else 0
    
    print(f"Average Diversity: IKA = {ika_avg_diversity:.2f}, ARBA = {arba_avg_diversity:.2f}" + 
          (f", anonypy = {anonypy_avg_diversity:.2f}" if anonypy_avg_diversity > 0 else ""))
    
    print(f"Diversity Improvement (ARBA vs IKA): {((arba_avg_diversity - ika_avg_diversity) / ika_avg_diversity * 100):.2f}%")
    if anonypy_avg_diversity > 0:
        print(f"Diversity Improvement (ARBA vs anonypy): {((arba_avg_diversity - anonypy_avg_diversity) / anonypy_avg_diversity * 100):.2f}%")
    else:
        print("Diversity Improvement (ARBA vs anonypy): N/A (anonypy failed)")
    
    # Verify algorithms satisfy their guarantees
    for name, group in ika_anonymized.groupby(qi_adult):
        assert len(group) >= 5, f"IKA k-anonymity violated for group {name}"
    
    for name, group in arba_anonymized.groupby(qi_adult):
        assert len(group) >= 5, f"ARBA k-anonymity violated for group {name}"
        assert group[sa_adult].nunique() >= 3, f"ARBA l-diversity violated for group {name}"

if __name__ == "__main__":
    test_adult_comparison()