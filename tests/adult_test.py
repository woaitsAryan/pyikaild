from pyikaild.ika import IKA
from pyikaild.ild import ILD
import pandas as pd

def test_adult():
    print("\n\n--- Example 3: Adult Dataset Snippet ---")
    # Load adult data (replace 'adult.data' with your path if needed)
    # Download from: https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
    # Add header row as it's missing in the original file
    adult_colnames = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
        'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
        'hours-per-week', 'native-country', 'income'
    ]
    adult_df_full = pd.read_csv(
        'adult.data',
        header=None,
        names=adult_colnames,
        na_values=' ?', # Handle missing values marked as ' ?'
        skipinitialspace=True
    )
    adult_df = adult_df_full.sample(n=1000, random_state=42).dropna() # Sample & drop NA for simplicity
    print(f"Using Adult dataset sample ({len(adult_df)} records)")
    
    # Define QIs and SA (Common choices)
    qi_adult = ['age', 'workclass', 'education', 'race', 'sex']
    sa_adult = 'occupation' # Sensitive attribute as per MPSEC example in paper
    
    # Specify types (important for Adult data)
    num_qi_adult = ['age']
    cat_qi_adult = ['workclass', 'education', 'race', 'sex']

    # Apply IKA (k=10)
    ika_adult = IKA(k=10,
                    qi_attributes=qi_adult,
                    sa_attribute=sa_adult,
                    numerical_qi=num_qi_adult,
                    categorical_qi=cat_qi_adult,
                    max_split_level=15) # Limit split level for faster example run
    
    anonymized_adult_df = ika_adult.fit_transform(adult_df)
    print("\nSample of IKA Anonymized Adult Data (k=10):")
    print(anonymized_adult_df.head())

    il_adult = ika_adult.get_information_loss()
    print(f"\nInformation Loss (IKA, Adult Data): {il_adult:.4f}" if il_adult is not None else "")

    # Apply ILD (l=5)
    ild_adult = ILD(l=5, qi_attributes=qi_adult, sa_attribute=sa_adult)
    diverse_adult_df = ild_adult.transform(anonymized_adult_df)
    print("\nSample of ILD Applied Adult Data (l=5):")
    print(diverse_adult_df.head())

    # Verify l-diversity (check a few groups)
    print("\nVerifying l=5 diversity (sample groups):")
    adult_groups = diverse_adult_df.groupby(qi_adult)
    groups_checked = 0
    for name, group in adult_groups:
        if groups_checked < 5: # Print first 5 groups found
            count = group[sa_adult].nunique()
            status = "OK" if count >= 5 else "VIOLATES"
            print(f"Group {name}: SA Count = {count} ({status})")
            groups_checked += 1
        else:
            break # Stop after checking a few

    # Add assertions
    assert anonymized_adult_df.shape == adult_df.shape, "Anonymized data shape mismatch"
    assert il_adult is not None and 0 <= il_adult <= 1, "Information loss is out of expected range"
    for name, group in adult_groups:
        assert group[sa_adult].nunique() >= 5, f"l-diversity violated for group {name}"

if __name__ == "__main__":
    test_adult()