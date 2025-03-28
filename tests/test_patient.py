from pyikaild.ika import IKA
from pyikaild.ild import ILD
import pandas as pd

def test_patient_data():
    print("\n--- Example 1: Patient Data ---")
    patient_data = {
        'Name': ['Anand', 'Bharti', 'Carl', 'Diana', 'Emily', 'Fatima', 'Garvin'],
        'Age': [45, 47, 52, 53, 64, 67, 62],
        'Pincode': [400052, 400058, 400032, 400045, 100032, 100053, 200045],
        'Job': ['Writer', 'Writer', 'Lawyer', 'Artist', 'Lawyer', 'Lawyer', 'Writer'],
        'Disease': ['Flu', 'Pneumonia', 'Flu', 'Stomach ulcers', 'Stomach infection', 'Hepatitis', 'Stomach cancer']
    }
    patient_df = pd.DataFrame(patient_data)
    print("\nOriginal Patient Data:")
    print(patient_df)

    # Define QIs and SA
    # Using Age and Pincode as QIs, Disease as SA
    # Note: Pincode is numeric here, Age is numeric. Job is non-sensitive identifier.
    qi_patient = ['Age', 'Pincode']
    sa_patient = 'Disease'
    num_qi_patient = ['Age', 'Pincode'] # Specify numerical QIs

    # Apply IKA (k=3 for this small example)
    ika_patient = IKA(k=3, qi_attributes=qi_patient, sa_attribute=sa_patient, numerical_qi=num_qi_patient)
    ika_patient.fit(patient_df.drop(columns=['Name'])) # Drop direct identifier
    anonymized_patient_df = ika_patient.transform(patient_df.drop(columns=['Name']))

    print("\nIKA Anonymized Patient Data (k=3):")
    print(anonymized_patient_df)

    # Calculate Information Loss
    il_patient = ika_patient.get_information_loss()
    print(f"\nInformation Loss (IKA, Patient Data): {il_patient:.4f}" if il_patient is not None else "")

    # Apply ILD (l=2) after IKA
    ild_patient = ILD(l=2, qi_attributes=qi_patient, sa_attribute=sa_patient)
    diverse_patient_df = ild_patient.transform(anonymized_patient_df)

    print("\nILD Applied Patient Data (l=2):")
    # Display relevant columns
    print(diverse_patient_df[['Age', 'Pincode', 'Disease']])

    # Verify l-diversity (manual check for small data)
    print("\nVerifying l=2 diversity:")
    for name, group in diverse_patient_df.groupby(qi_patient):
        print(f"Group {name}: SA values = {list(group[sa_patient].unique())}, Count = {group[sa_patient].nunique()}")

    # Add assertions
    assert anonymized_patient_df.shape == patient_df.drop(columns=['Name']).shape, "Anonymized data shape mismatch"
    assert il_patient is not None and 0 <= il_patient <= 1, "Information loss is out of expected range"
    for name, group in diverse_patient_df.groupby(qi_patient):
        assert group[sa_patient].nunique() >= 2, f"l-diversity violated for group {name}"
