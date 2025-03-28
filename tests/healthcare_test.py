from pyikaild.ild import ILD
import pandas as pd

def test_healthcare():
    print("\n\n--- Example 2: Healthcare Data ---")
    # Simulating k-anonymized data similar to Table 3 structure
    healthcare_data_anon = {
        'ZipCode': ['130**', '130**', '130**', '130**', '130**', '130**', '130**', '130**', '130**', '130**'],
        'Age': ['< 30', '< 30', '< 30', '< 30', '< 30', '3*', '3*', '3*', '3*', '3*'],
        'Nationality': ['*', '*', '*', '*', '*', '*', '*', '*', '*', '*'],
        'Disease': ['Cancer', 'Cancer', 'Corona', 'Cancer', 'Cancer', 'Heart Disease', 'Heart Disease', 'Cancer', 'Cancer', 'Corona'] # SA
    }
    healthcare_df_anon = pd.DataFrame(healthcare_data_anon)
    print("\nInput (Simulated Anonymized) Healthcare Data:")
    print(healthcare_df_anon)

    # Define QIs and SA for this dataset
    qi_health = ['ZipCode', 'Age', 'Nationality'] # All seem generalized/suppressed
    sa_health = 'Disease'

    # Apply ILD (l=3, as suggested by the transition from Table 3 to 4)
    ild_health = ILD(l=3, qi_attributes=qi_health, sa_attribute=sa_health)
    diverse_health_df = ild_health.transform(healthcare_df_anon)

    print(f"\nILD Applied Healthcare Data (l=3):")
    print(diverse_health_df)

    # Verify l-diversity
    print("\nVerifying l=3 diversity:")
    for name, group in diverse_health_df.groupby(qi_health):
        print(f"Group {name}: SA values = {list(group[sa_health].unique())}, Count = {group[sa_health].nunique()}")

    # Add assertions
    assert diverse_health_df.shape == healthcare_df_anon.shape, "Transformed data shape mismatch"
    for name, group in diverse_health_df.groupby(qi_health):
        assert group[sa_health].nunique() >= 3, f"l-diversity violated for group {name}"
