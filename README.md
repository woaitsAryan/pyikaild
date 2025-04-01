# pyikaild

A Python package for data privacy and anonymization implementing Improved k-Anonymity (IKA) and Improved l-Diversity (ILD) algorithms.

## Overview

`pyikaild` provides implementations of two key privacy-preserving techniques for sensitive data:

1. **Improved k-Anonymity (IKA)**: Ensures each record is indistinguishable from at least k-1 other records based on quasi-identifier attributes through generalization techniques.

2. **Improved l-Diversity (ILD)**: Ensures each equivalence class (group of records with identical quasi-identifiers) contains at least l distinct values for sensitive attributes.

These techniques help protect privacy in datasets while maintaining data utility for analysis, in compliance with privacy regulations and best practices.

## Installation

```bash
pip install pyikaild
```

## Key Concepts

- **Quasi-Identifiers (QI)**: Attributes that, when combined, could potentially identify an individual (e.g., age, zip code, gender)
- **Sensitive Attribute (SA)**: Data that should be protected (e.g., disease, salary)
- **k-Anonymity**: Each record is indistinguishable from at least k-1 other records
- **l-Diversity**: Each group of records with identical QIs has at least l different values for sensitive attributes

## Usage

### Basic Example

```python
from pyikaild.ika import IKA
from pyikaild.ild import ILD
import pandas as pd

# Sample dataset
data = {
    'Age': [45, 47, 52, 53, 64, 67, 62],
    'Zipcode': [400052, 400058, 400032, 400045, 100032, 100053, 200045],
    'Disease': ['Flu', 'Pneumonia', 'Flu', 'Stomach ulcers', 'Stomach infection', 'Hepatitis', 'Stomach cancer']
}
df = pd.DataFrame(data)

# Define quasi-identifiers and sensitive attribute
qi_attributes = ['Age', 'Zipcode']
sa_attribute = 'Disease'
numerical_qi = ['Age', 'Zipcode']  # Specify which QIs are numerical

# Apply k-anonymity (k=3)
ika = IKA(k=3, 
          qi_attributes=qi_attributes, 
          sa_attribute=sa_attribute, 
          numerical_qi=numerical_qi)
anonymized_df = ika.fit_transform(df)

# Calculate information loss
info_loss = ika.get_information_loss()
print(f"Information Loss: {info_loss:.4f}")

# Apply l-diversity (l=2) on the k-anonymized data
ild = ILD(l=2, qi_attributes=qi_attributes, sa_attribute=sa_attribute)
diverse_df = ild.transform(anonymized_df)

# Verify l-diversity
for name, group in diverse_df.groupby(qi_attributes):
    print(f"Group {name}: SA count = {group[sa_attribute].nunique()}")
```

### Adult Dataset Example

```python
# Load adult dataset (available from UCI ML Repository)
adult_colnames = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
    'hours-per-week', 'native-country', 'income'
]
adult_df = pd.read_csv('adult.data', header=None, names=adult_colnames, 
                      na_values=' ?', skipinitialspace=True)

# Define QIs and SA
qi_adult = ['age', 'workclass', 'education', 'race', 'sex']
sa_adult = 'occupation'
num_qi_adult = ['age']
cat_qi_adult = ['workclass', 'education', 'race', 'sex']

# Apply IKA (k=10)
ika_adult = IKA(k=10,
                qi_attributes=qi_adult,
                sa_attribute=sa_adult,
                numerical_qi=num_qi_adult,
                categorical_qi=cat_qi_adult,
                max_split_level=8)

anonymized_adult_df = ika_adult.fit_transform(adult_df)

# Apply ILD (l=5)
ild_adult = ILD(l=5, qi_attributes=qi_adult, sa_attribute=sa_adult)
diverse_adult_df = ild_adult.transform(anonymized_adult_df)
```

## API Reference

### IKA (Improved k-Anonymization)

```python
class IKA:
    def __init__(self, k, qi_attributes, sa_attribute, numerical_qi=None, 
                 categorical_qi=None, max_split_level=10):
        """
        Parameters:
        -----------
        k : int
            The minimum size of an equivalence class (>= 2)
        qi_attributes : List[str]
            List of column names to be treated as Quasi-Identifiers
        sa_attribute : str
            Column name of the Sensitive Attribute
        numerical_qi : List[str], optional
            List of QI attributes that are numerical
        categorical_qi : List[str], optional
            List of QI attributes that are categorical
        max_split_level : int, default=10
            Maximum recursion depth for splitting (controls granularity)
        """
        
    def fit(self, df):
        """Fit the model to the DataFrame, partitioning it for k-anonymity"""
        
    def transform(self, df):
        """Transform the DataFrame to achieve k-anonymity"""
        
    def fit_transform(self, df):
        """Fit and transform in one step"""
        
    def get_information_loss(self):
        """Calculate information loss due to anonymization"""
```

### ILD (Improved l-Diversity)

```python
class ILD:
    def __init__(self, l, qi_attributes, sa_attribute):
        """
        Parameters:
        -----------
        l : int
            The minimum number of distinct sensitive values required per group (>= 2)
        qi_attributes : List[str]
            List of column names treated as Quasi-Identifiers (should match those used in IKA)
        sa_attribute : str
            Column name of the Sensitive Attribute
        """
        
    def transform(self, df):
        """Apply l-diversity enforcement to a k-anonymized DataFrame"""
        
    def fit_transform(self, df):
        """Transform the data to enforce l-diversity (fit is not needed)"""
```

## Algorithm Details

### IKA Algorithm
1. Recursively partition the dataset based on QI attributes
2. Ensure each partition has at least k records
3. Generalize QI values within each partition
   - Numerical: Represented as ranges [min-max]
   - Categorical: Set to common value or '*' if values differ

### ILD Algorithm
1. Identify equivalence classes that violate l-diversity
2. Borrow sensitive attribute values from other diverse classes
3. Modify records in violating classes to ensure l-diversity
4. Verify the result satisfies l-diversity constraints

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)
