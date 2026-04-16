import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy import stats

# Create output directory
output_dir = os.path.join(os.getcwd(), "results")
os.makedirs(output_dir, exist_ok=True)

# Parameter settings
c = 1.0  # constant
A_values = [1, 5, 10, 20, 50, 100]  # area gradient
t_max = 1000  # total time
t = np.linspace(0, 1000, t_max)  # time axis

# Process parameters - baseline intensities set as constants
D0_A = 1      # diffusion baseline intensity
E0_A = 0.8    # extinction baseline intensity
Sp0_A = 0.01  # speciation baseline intensity

ω_d, ω_e, ω_s = 0.1, 0.05, 0.01  # fluctuation frequencies
τ_d = 0.5   # diffusion delay coefficient
ϕ_e = 0.5   # extinction phase coefficient
φ_s = 0.1   # speciation phase coefficient
α_e = 0.8   # extinction amplitude

# Define three process functions (without random noise)
def D(A, t):
    return D0_A * (1 + np.sin(ω_d * t - τ_d * np.log(A)))

def E(A, t):
    return E0_A * (1 + α_e * np.sin(ω_e * t + ϕ_e * np.log(A)))

def Sp(A, t):
    return Sp0_A * (1 + np.sin(ω_s * t + φ_s * np.log(A)))

# Define z(A,t) function
def z(A, t):
    return D(A, t) + Sp(A, t) - E(A, t)

# Define species richness function
def S(A, t, c=1.0):
    z_val = z(A, t)
    return c * (A ** z_val)

# Use all time points as samples
sample_times = t  # all 1000 time points

# Calculate z-value characteristics at each time point
results = []
for i, ti in enumerate(sample_times):
    # Species numbers for each area at current time
    S_vals = [S(A, ti) for A in A_values]
    
    # z-values for each area at current time
    z_vals = [z(A, ti) for A in A_values]
    
    # Statistical characteristics of z-values
    z_mean = np.mean(z_vals)
    z_std = np.std(z_vals)
    z_min = np.min(z_vals)
    z_max = np.max(z_vals)
    
    # Characteristics of the three processes
    D_vals = [D(A, ti) for A in A_values]
    E_vals = [E(A, ti) for A in A_values]
    Sp_vals = [Sp(A, ti) for A in A_values]
    
    # Calculate SAR pattern significance (p-value)
    # Use linear regression to test relationship between log(S) and log(A)
    log_A = np.log(A_values)
    log_S = np.log(S_vals)
    
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_A, log_S)
    
    # Determine SAR type
    if p_value < 0.05:  # statistically significant
        if slope > 0:   # positive correlation
            sar_type = "Positive SAR"
        else:           # negative correlation
            sar_type = "Negative SAR"
    else:               # not significant
        sar_type = "Non-SAR"
    
    # Save species counts per area
    species_by_area = {f"S_A{A}": S_val for A, S_val in zip(A_values, S_vals)}
    
    results.append({
        'time': ti,
        'z_mean': z_mean,
        'z_std': z_std,
        'z_min': z_min,
        'z_max': z_max,
        'D_mean': np.mean(D_vals),
        'D_std': np.std(D_vals),
        'E_mean': np.mean(E_vals),
        'E_std': np.std(E_vals),
        'Sp_mean': np.mean(Sp_vals),
        'Sp_std': np.std(Sp_vals),
        'sar_type': sar_type,
        'slope': slope,
        'p_value': p_value,
        'r_squared': r_value**2,
        **species_by_area   # add species counts for each area
    })

# Convert to DataFrame
df = pd.DataFrame(results)

# Define strict outlier removal function – ensure complete removal of non‑conforming points
def remove_all_outliers(data, columns_to_check, group_by='sar_type'):
    """
    Strictly remove all outliers to ensure boxplots have no outliers.
    """
    filtered_data = data.copy()
    
    for column in columns_to_check:
        for sar_type in filtered_data[group_by].unique():
            group_mask = filtered_data[group_by] == sar_type
            group_data = filtered_data[group_mask]
            
            if len(group_data) == 0:
                continue
                
            # Compute IQR
            Q1 = group_data[column].quantile(0.25)
            Q3 = group_data[column].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier boundaries
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Mark outliers
            outlier_mask = (filtered_data[column] < lower_bound) | (filtered_data[column] > upper_bound)
            type_outlier_mask = outlier_mask & group_mask
            
            # Remove outliers
            filtered_data = filtered_data[~type_outlier_mask]
    
    return filtered_data

# Define strict SAR type validation function
def validate_sar_type(data):
    """
    Ensure strict validity of SAR type classification.
    """
    validated_data = data.copy()
    
    for idx, row in validated_data.iterrows():
        p_value = row['p_value']
        slope = row['slope']
        r_squared = row['r_squared']
        
        # Strict validation criteria
        is_valid_positive = (p_value < 0.05) and (slope > 0) and (r_squared > 0.5)
        is_valid_negative = (p_value < 0.05) and (slope < 0) and (r_squared > 0.5)
        is_valid_non = (p_value >= 0.05) or (r_squared <= 0.5)
        
        # Reassign SAR type
        if is_valid_positive:
            validated_data.at[idx, 'sar_type'] = "Positive SAR"
        elif is_valid_negative:
            validated_data.at[idx, 'sar_type'] = "Negative SAR"
        else:
            validated_data.at[idx, 'sar_type'] = "Non-SAR"
    
    return validated_data

print("Original data size:", len(df))

# Step 1: Strict SAR type validation
df_validated = validate_sar_type(df)
print("After strict SAR validation, data size:", len(df_validated))

# Step 2: Strict outlier removal
columns_to_check = ['z_mean', 'slope', 'r_squared', 'D_mean', 'E_mean', 'Sp_mean']
df_clean = remove_all_outliers(df_validated, columns_to_check)
print("After strict outlier removal, data size:", len(df_clean))

# Step 3: Verify that filtered data indeed has no outliers
def verify_no_outliers(data, columns_to_check, group_by='sar_type'):
    """
    Verify that the data actually contains no outliers.
    """
    has_outliers = False
    
    for column in columns_to_check:
        for sar_type in data[group_by].unique():
            group_data = data[data[group_by] == sar_type][column]
            
            if len(group_data) == 0:
                continue
                
            # Compute IQR
            Q1 = group_data.quantile(0.25)
            Q3 = group_data.quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier boundaries
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Check for outliers
            outliers = group_data[(group_data < lower_bound) | (group_data > upper_bound)]
            
            if len(outliers) > 0:
                print(f"Warning: {sar_type} still has {len(outliers)} outliers in column {column}")
                has_outliers = True
    
    return not has_outliers

# Verify cleaning effectiveness
is_clean = verify_no_outliers(df_clean, columns_to_check)
print(f"Data cleaning verification: {'Passed' if is_clean else 'Failed'}")

# Extract three SAR types (using strictly cleaned data)
positive_sar = df_clean[df_clean['sar_type'] == "Positive SAR"]
negative_sar = df_clean[df_clean['sar_type'] == "Negative SAR"]
non_sar = df_clean[df_clean['sar_type'] == "Non-SAR"]

# Compute proportions and counts (using strictly cleaned data)
total_samples = len(df_clean)
positive_count = len(positive_sar)
negative_count = len(negative_sar)
non_count = len(non_sar)

positive_ratio = positive_count / total_samples if total_samples > 0 else 0
negative_ratio = negative_count / total_samples if total_samples > 0 else 0
non_ratio = non_count / total_samples if total_samples > 0 else 0

# Create ratio statistics DataFrame
ratio_stats = pd.DataFrame({
    'SAR Type': ['Positive SAR', 'Negative SAR', 'Non-SAR'],
    'Count': [positive_count, negative_count, non_count],
    'Ratio': [positive_ratio, negative_ratio, non_ratio]
})

# Calculate statistics for each SAR type
def calculate_statistics(data, sar_type):
    if len(data) == 0:
        return pd.DataFrame()
    
    stats = {
        'SAR Type': sar_type,
        'Count': len(data),
        'z_mean_mean': data['z_mean'].mean(),
        'z_mean_std': data['z_mean'].std(),
        'slope_mean': data['slope'].mean(),
        'slope_std': data['slope'].std(),
        'p_value_mean': data['p_value'].mean(),
        'p_value_std': data['p_value'].std(),
        'r_squared_mean': data['r_squared'].mean(),
        'r_squared_std': data['r_squared'].std(),
        'D_mean_mean': data['D_mean'].mean(),
        'D_mean_std': data['D_mean'].std(),
        'E_mean_mean': data['E_mean'].mean(),
        'E_mean_std': data['E_mean'].std(),
        'Sp_mean_mean': data['Sp_mean'].mean(),
        'Sp_mean_std': data['Sp_mean'].std()
    }
    return pd.DataFrame([stats])

# Compute statistics
stats_positive = calculate_statistics(positive_sar, "Positive SAR")
stats_negative = calculate_statistics(negative_sar, "Negative SAR")
stats_non = calculate_statistics(non_sar, "Non-SAR")

# Combine statistics
all_stats = pd.concat([stats_positive, stats_negative, stats_non], ignore_index=True)

# Save results (original and strictly cleaned data)
all_stats.to_csv(os.path.join(output_dir, 'sar_statistics_strictly_cleaned.csv'), index=False)
positive_sar.to_csv(os.path.join(output_dir, 'positive_sar_samples_strictly_cleaned.csv'), index=False)
negative_sar.to_csv(os.path.join(output_dir, 'negative_sar_samples_strictly_cleaned.csv'), index=False)
non_sar.to_csv(os.path.join(output_dir, 'non_sar_samples_strictly_cleaned.csv'), index=False)
ratio_stats.to_csv(os.path.join(output_dir, 'sar_ratios_strictly_cleaned.csv'), index=False)
df_clean.to_csv(os.path.join(output_dir, 'all_samples_strictly_cleaned.csv'), index=False)

# Also save original data for comparison
df.to_csv(os.path.join(output_dir, 'all_samples_original.csv'), index=False)

# Plot boxplots after strict cleaning
def plot_strictly_cleaned_boxplots(df_clean, output_dir):
    """Plot boxplots after strict cleaning."""
    
    # Extract three SAR types
    positive_sar = df_clean[df_clean['sar_type'] == "Positive SAR"]
    negative_sar = df_clean[df_clean['sar_type'] == "Negative SAR"]
    non_sar = df_clean[df_clean['sar_type'] == "Non-SAR"]
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # z-value distribution
    if len(positive_sar) > 0 and len(negative_sar) > 0 and len(non_sar) > 0:
        axes[0,0].boxplot([positive_sar['z_mean'], negative_sar['z_mean'], non_sar['z_mean']], 
                         labels=['Positive SAR', 'Negative SAR', 'Non-SAR'])
    axes[0,0].set_ylabel('z value')
    axes[0,0].set_title('Distribution of z values (Strictly Cleaned)')
    axes[0,0].grid(True)
    
    # Slope distribution
    if len(positive_sar) > 0 and len(negative_sar) > 0 and len(non_sar) > 0:
        axes[0,1].boxplot([positive_sar['slope'], negative_sar['slope'], non_sar['slope']], 
                         labels=['Positive SAR', 'Negative SAR', 'Non-SAR'])
    axes[0,1].set_ylabel('Slope')
    axes[0,1].set_title('Distribution of slopes (Strictly Cleaned)')
    axes[0,1].grid(True)
    
    # R² distribution
    if len(positive_sar) > 0 and len(negative_sar) > 0 and len(non_sar) > 0:
        axes[0,2].boxplot([positive_sar['r_squared'], negative_sar['r_squared'], non_sar['r_squared']], 
                         labels=['Positive SAR', 'Negative SAR', 'Non-SAR'])
    axes[0,2].set_ylabel('R²')
    axes[0,2].set_title('Distribution of R² (Strictly Cleaned)')
    axes[0,2].grid(True)
    
    # Distribution of the three processes
    # Diffusion process
    if len(positive_sar) > 0 and len(negative_sar) > 0 and len(non_sar) > 0:
        axes[1,0].boxplot([positive_sar['D_mean'], negative_sar['D_mean'], non_sar['D_mean']], 
                         labels=['Positive SAR', 'Negative SAR', 'Non-SAR'])
    axes[1,0].set_ylabel('D value')
    axes[1,0].set_title('Distribution of Diffusion (Strictly Cleaned)')
    axes[1,0].grid(True)
    
    # Extinction process
    if len(positive_sar) > 0 and len(negative_sar) > 0 and len(non_sar) > 0:
        axes[1,1].boxplot([positive_sar['E_mean'], negative_sar['E_mean'], non_sar['E_mean']], 
                         labels=['Positive SAR', 'Negative SAR', 'Non-SAR'])
    axes[1,1].set_ylabel('E value')
    axes[1,1].set_title('Distribution of Extinction (Strictly Cleaned)')
    axes[1,1].grid(True)
    
    # Speciation process
    if len(positive_sar) > 0 and len(negative_sar) > 0 and len(non_sar) > 0:
        axes[1,2].boxplot([positive_sar['Sp_mean'], negative_sar['Sp_mean'], non_sar['Sp_mean']], 
                         labels=['Positive SAR', 'Negative SAR', 'Non-SAR'])
    axes[1,2].set_ylabel('Sp value')
    axes[1,2].set_title('Distribution of Speciation (Strictly Cleaned)')
    axes[1,2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'strictly_cleaned_distributions.png'), dpi=300)
    plt.show()

# Plot strictly cleaned boxplots
plot_strictly_cleaned_boxplots(df_clean, output_dir)

# Plot pie chart of SAR type proportions (strictly cleaned data)
plt.figure(figsize=(8, 8))
labels = ['Positive SAR', 'Negative SAR', 'Non-SAR']
sizes = [positive_count, negative_count, non_count]
colors_pie = ['lightgreen', 'lightcoral', 'lightblue']
explode = (0.1, 0, 0)  # explode Positive SAR

if sum(sizes) > 0:
    plt.pie(sizes, explode=explode, labels=labels, colors=colors_pie, autopct='%1.1f%%', 
             shadow=True, startangle=90)
else:
    plt.text(0.5, 0.5, 'No data after strict cleaning', horizontalalignment='center', 
             verticalalignment='center', transform=plt.gca().transAxes)

plt.axis('equal')
plt.title('Proportion of Different SAR Types (After Strict Cleaning)')
plt.savefig(os.path.join(output_dir, 'sar_proportions_strictly_cleaned.png'))
plt.show()

# Output statistics
print("\n=== Strict Data Cleaning Results ===")
print(f"Original data size: {len(df)}")
print(f"After strict SAR validation, data size: {len(df_validated)}")
print(f"After strict outlier removal, data size: {len(df_clean)}")
print(f"Data cleaning verification: {'Passed' if is_clean else 'Failed'}")

print("\n=== Proportions and counts of three SAR types (strictly cleaned) ===")
print(ratio_stats)
print(f"\nPositive SAR samples: {len(positive_sar)}")
print(f"Negative SAR samples: {len(negative_sar)}")
print(f"Non-SAR samples: {len(non_sar)}")

print("\n=== Detailed statistics (strictly cleaned) ===")
print(all_stats)

# Additional verification: check statistical characteristics of each SAR type
print("\n=== Statistical characteristics of each SAR type ===")
for sar_type in ['Positive SAR', 'Negative SAR', 'Non-SAR']:
    type_data = df_clean[df_clean['sar_type'] == sar_type]
    if len(type_data) > 0:
        print(f"\n{sar_type}:")
        print(f"  Sample count: {len(type_data)}")
        print(f"  z_mean range: [{type_data['z_mean'].min():.4f}, {type_data['z_mean'].max():.4f}]")
        print(f"  slope range: [{type_data['slope'].min():.4f}, {type_data['slope'].max():.4f}]")
        print(f"  R² range: [{type_data['r_squared'].min():.4f}, {type_data['r_squared'].max():.4f}]")

print("\nAnalysis complete! Strictly cleaned results saved to", output_dir)