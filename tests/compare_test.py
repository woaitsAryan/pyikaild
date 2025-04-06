# tests/test_arba_comparison.py
from pyikaild.ika import IKA, ARBA
import pandas as pd
import numpy as np
import time
import warnings
import anonypy
import csv

def test_arba_comparison():
    print("\n--- Comparing ARBA vs IKA vs anonypy on Patient Data ---")
    csv_file_path = './base_data.csv'

    patient_df = pd.read_csv(csv_file_path)
    print("\nOriginal Patient Data:")
    print(patient_df)

    # Define QIs and SA
    qi_patient = ['Age', 'Pincode']
    sa_patient = 'Disease'
    num_qi_patient = ['Age', 'Pincode']
    
    # Remove direct identifiers
    patient_df_clean = patient_df.drop(columns=['Name'])
    
    # 1. Apply IKA with k=3
    print("\n=== Testing IKA (k=3) ===")
    start_time = time.time()
    ika_patient = IKA(k=3, 
                      qi_attributes=qi_patient, 
                      sa_attribute=sa_patient, 
                      numerical_qi=num_qi_patient)
    
    ika_anonymized = ika_patient.fit_transform(patient_df_clean)
    ika_time = time.time() - start_time
    ika_loss = ika_patient.get_information_loss()
    
    print(f"\nIKA Execution Time: {ika_time:.4f} seconds")
    print(f"IKA Information Loss: {ika_loss:.4f}")
    print("\nIKA Anonymized Sample:")
    print(ika_anonymized.head())
    
    # Get diversity info from IKA
    ika_diversity = {}
    for name, group in ika_anonymized.groupby(qi_patient):
        ika_diversity[name] = group[sa_patient].nunique()
    
    print("\nIKA Diversity by group:")
    for name, diversity in ika_diversity.items():
        print(f"Group {name}: {diversity} distinct values")
    
    # 2. Apply ARBA with base_k=3 and diversity_threshold=2
    print("\n=== Testing ARBA (base_k=3, diversity_threshold=2) ===")
    start_time = time.time()
    arba_patient = ARBA(base_k=3, 
                        diversity_threshold=2,
                        qi_attributes=qi_patient, 
                        sa_attribute=sa_patient, 
                        numerical_qi=num_qi_patient)
    
    arba_anonymized = arba_patient.fit_transform(patient_df_clean)
    arba_time = time.time() - start_time
    arba_loss = arba_patient.get_information_loss()
    
    print(f"\nARBA Execution Time: {arba_time:.4f} seconds")
    print(f"ARBA Information Loss: {arba_loss:.4f}")
    print("\nARBA Anonymized Sample:")
    print(arba_anonymized.head())
    
    # Get diversity info from ARBA
    arba_diversity = {}
    for name, group in arba_anonymized.groupby(qi_patient):
        arba_diversity[name] = group[sa_patient].nunique()
    
    print("\nARBA Diversity by group:")
    for name, diversity in arba_diversity.items():
        print(f"Group {name}: {diversity} distinct values")
    
    # 3. anonypy comparison
    anonypy_time = None
    anonypy_info_loss = None
    anonypy_diversity = {}
    anonypy_avg_diversity = 0
    
    print("\n=== Testing anonypy (k=3, l=2) ===")
    
    # Create a copy of the dataset for anonypy
    anonypy_df = patient_df_clean.copy()
    
    # Convert to proper format for anonypy
    # Note: anonypy expects numerical columns to be numerical and categorical to be 'category' dtype
    for col in num_qi_patient:
        anonypy_df[col] = pd.to_numeric(anonypy_df[col])
    
    # Exclude Job column which is not in QI or SA for simplicity
    anonypy_df = anonypy_df[qi_patient + [sa_patient]]
    
    # Initialize the Preserver with our dataset
    start_time = time.time()
    
    try:
        preserver = anonypy.Preserver(anonypy_df, qi_patient, sa_patient)
        
        # Get l-diversity (k=3, l=2)
        anonypy_result = preserver.anonymize_l_diversity(k=3, l=2)
        
        anonypy_time = time.time() - start_time
        
        # Create a DataFrame from the result rows
        anonypy_anonymized = pd.DataFrame(anonypy_result)
        
        print(f"\nanonypy Execution Time: {anonypy_time:.4f} seconds")
        
        # Estimate information loss (based on generalization extent)
        # Use range width as a proxy for information loss
        anonypy_info_loss = 0.0
        
        # Calculate diversity for each generalized group
        unique_qis = set()
        for row in anonypy_result:
            # Convert any list or unhashable values to tuples
            qi_values = []
            for col in qi_patient:
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
                for i, col in enumerate(qi_patient):
                    val = row[col]
                    if isinstance(val, list) or not isinstance(val, (str, int, float, tuple)):
                        val = str(val)
                    row_qi_values.append(val)
                
                if tuple(row_qi_values) == qi_key:
                    sensitive_values.add(row[sa_patient])
            anonypy_diversity[qi_key] = len(sensitive_values)
        
        print("\nanonypy Anonymized Result:")
        print(pd.DataFrame(anonypy_result).head())
        
        print("\nanonypy Diversity by group:")
        for qi_key, diversity in anonypy_diversity.items():
            print(f"Group {qi_key}: {diversity} distinct values")
            
        anonypy_avg_diversity = sum(anonypy_diversity.values()) / len(anonypy_diversity) if anonypy_diversity else 0
        
        # Estimate information loss based on generalization ranges
        # For numeric attributes, calculate average range size relative to original
        info_loss_sum = 0.0
        features_count = 0
        
        for attr in qi_patient:
            if attr in num_qi_patient:
                original_range = anonypy_df[attr].max() - anonypy_df[attr].min()
                feature_loss = 0.0
                weight = 0
                
                for row in anonypy_result:
                    count = row.get('count', 1)
                    val = row[attr]
                    try:
                        if '-' in str(val):
                            min_val, max_val = map(float, str(val).split('-'))
                            range_size = max_val - min_val
                            feature_loss += (range_size / original_range) * count
                        else:
                            # Point value has no generalization loss
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
    for name, group in ika_anonymized.groupby(qi_patient):
        assert len(group) >= 3, f"IKA k-anonymity violated for group {name}"
    
    for name, group in arba_anonymized.groupby(qi_patient):
        assert len(group) >= 3, f"ARBA k-anonymity violated for group {name}"
        assert group[sa_patient].nunique() >= 2, f"ARBA l-diversity violated for group {name}"

if __name__ == "__main__":
    test_arba_comparison()