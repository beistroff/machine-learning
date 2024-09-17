import pandas as pd
import seaborn as sns
from scipy.stats import skew, kurtosis, trim_mean
from statsmodels import robust
import numpy as np

df = sns.load_dataset('iris')

def describe_extended(df):
    summary = df.describe().T
    summary['vars'] = range(1, len(summary) + 1)                    # Vars index
    summary['mad'] = df.apply(robust.mad)                           # Median Absolute Deviation
    summary['skew'] = df.apply(skew)                                # Skewness
    summary['kurtosis'] = df.apply(kurtosis)                        # Kurtosis
    summary['trimmed_mean'] = df.apply(lambda x: trim_mean(x, 0.1)) # Trimmed Mean
    summary['se'] = summary['std'] / np.sqrt(summary['count'])      # Standard Error
    summary['range'] = summary['max'] - summary['min']              # Range
    return summary[['vars', 'count', 'mean', 'std', '50%', 'trimmed_mean', 'mad', 
                    'min', 'max', 'range', 'skew', 'kurtosis', 'se']]

# Filter only numeric columns
numeric_df = df.select_dtypes(include=[np.number])

# Get the extended description
extended_summary = describe_extended(numeric_df)

# Print extended summary
print(extended_summary.round(2))
