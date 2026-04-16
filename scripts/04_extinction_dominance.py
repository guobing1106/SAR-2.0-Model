# script4_analysis_modified.py
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

# Initial species pool size
initial_species_pool = 1000

# Process parameters – fixed baseline values from the original script
D0_A = 0   # diffusion baseline set to 0
E0_A = 1   # extinction baseline set to 1 (fixed)
Sp0_A = 0  # speciation baseline set to 0

# Fluctuation parameters
ω_d, ω_e, ω_s = 0.1, 0.05, 0.01  # fluctuation frequencies
τ_d = 0.5   # diffusion delay coefficient
ϕ_e = 0.5   # extinction phase coefficient
φ_s = 0.1   # speciation phase coefficient
α_e = 0.8   # extinction amplitude

# Define the three process functions (extinction baseline fixed)
def D(A, t):
    return D0_A * (1 + np.sin(ω_d * t - τ_d * np.log(A)))

def E(A, t):
    # Fixed extinction baseline, does not vary with area
    return E0_A * (1 + α_e * np.sin(ω_e * t + ϕ_e * np.log(A)))

def Sp(A, t):
    return Sp0_A * (1 + np.sin(ω_s * t + φ_s * np.log(A)))

# Define z(A,t) function
def z(A, t):
    return D(A, t) + Sp(A, t) - E(A, t)

# Modified species richness function: extinction fluctuations on initial species pool
def S(A, t, c=1.0):
    # Initial species pool = 1000, affected by extinction process
    extinction_effect = E(A, t)
    
    # Use cumulative extinction effect, not instantaneous
    cumulative_extinction = np.minimum(1.0, extinction_effect * 0.001 * t)
    
    # Compute survival rate, ensure non-negative
    survival_rate = np.maximum(0, 1 - cumulative_extinction)
    
    # Minimum species count is 0
    min_species = 0
    
    return np.maximum(min_species, initial_species_pool * survival_rate)

# Use all time points as samples
sample_times = t  # all 1000 time points

# Compute z-value characteristics at each time point
results = []
for i, ti in enumerate(sample_times):
    # Species counts for each area at current time
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
    # Linear regression to test relationship between log(S) and log(A)
    log_A = np.log(A_values)
    log_S = np.log([max(0.001, s) for s in S_vals])  # avoid log(0)
    
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

# Define strict outlier removal function
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

# -------------------------------------------------------------------
# Plot 1: z-value distribution boxplot (separate figure)
# -------------------------------------------------------------------
if len(positive_sar) > 0 and len(negative_sar) > 0 and len(non_sar) > 0:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot([positive_sar['z_mean'], negative_sar['z_mean'], non_sar['z_mean']],
               labels=['Positive SAR', 'Negative SAR', 'Non-SAR'])
    ax.set_ylabel('z value')
    ax.set_title('Distribution of z values (Strictly Cleaned)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'boxplot_z_values.png'), dpi=300)
    plt.close(fig)

# -------------------------------------------------------------------
# Plot 2: slope distribution boxplot (separate figure)
# -------------------------------------------------------------------
if len(positive_sar) > 0 and len(negative_sar) > 0 and len(non_sar) > 0:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot([positive_sar['slope'], negative_sar['slope'], non_sar['slope']],
               labels=['Positive SAR', 'Negative SAR', 'Non-SAR'])
    ax.set_ylabel('Slope')
    ax.set_title('Distribution of slopes (Strictly Cleaned)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'boxplot_slopes.png'), dpi=300)
    plt.close(fig)

# -------------------------------------------------------------------
# Plot 3: R² distribution boxplot (separate figure)
# -------------------------------------------------------------------
if len(positive_sar) > 0 and len(negative_sar) > 0 and len(non_sar) > 0:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot([positive_sar['r_squared'], negative_sar['r_squared'], non_sar['r_squared']],
               labels=['Positive SAR', 'Negative SAR', 'Non-SAR'])
    ax.set_ylabel('R²')
    ax.set_title('Distribution of R² (Strictly Cleaned)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'boxplot_rsquared.png'), dpi=300)
    plt.close(fig)

# -------------------------------------------------------------------
# Plot 4: Diffusion mean distribution boxplot (separate figure)
# -------------------------------------------------------------------
if len(positive_sar) > 0 and len(negative_sar) > 0 and len(non_sar) > 0:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot([positive_sar['D_mean'], negative_sar['D_mean'], non_sar['D_mean']],
               labels=['Positive SAR', 'Negative SAR', 'Non-SAR'])
    ax.set_ylabel('D value')
    ax.set_title('Distribution of Diffusion (Strictly Cleaned)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'boxplot_diffusion.png'), dpi=300)
    plt.close(fig)

# -------------------------------------------------------------------
# Plot 5: Extinction mean distribution boxplot (separate figure)
# -------------------------------------------------------------------
if len(positive_sar) > 0 and len(negative_sar) > 0 and len(non_sar) > 0:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot([positive_sar['E_mean'], negative_sar['E_mean'], non_sar['E_mean']],
               labels=['Positive SAR', 'Negative SAR', 'Non-SAR'])
    ax.set_ylabel('E value')
    ax.set_title('Distribution of Extinction (Strictly Cleaned)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'boxplot_extinction.png'), dpi=300)
    plt.close(fig)

# -------------------------------------------------------------------
# Plot 6: Speciation mean distribution boxplot (separate figure)
# -------------------------------------------------------------------
if len(positive_sar) > 0 and len(negative_sar) > 0 and len(non_sar) > 0:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot([positive_sar['Sp_mean'], negative_sar['Sp_mean'], non_sar['Sp_mean']],
               labels=['Positive SAR', 'Negative SAR', 'Non-SAR'])
    ax.set_ylabel('Sp value')
    ax.set_title('Distribution of Speciation (Strictly Cleaned)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'boxplot_speciation.png'), dpi=300)
    plt.close(fig)

# -------------------------------------------------------------------
# Plot 7: Pie chart of SAR type proportions
# -------------------------------------------------------------------
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
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'sar_proportions_strictly_cleaned.png'), dpi=300)
plt.close()

# -------------------------------------------------------------------
# New: SAR power law visualizations (split into separate figures)
# -------------------------------------------------------------------
def plot_sar_power_law(df_clean, output_dir):
    """Plot SAR power law visualizations as separate figures."""

    # Select representative time points
    sample_times = [0, 250, 500, 750, 999]
    colors = plt.cm.viridis(np.linspace(0, 1, len(sample_times)))

    # Figure 1: Log-log plot of SAR at different times
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    for i, ti in enumerate(sample_times):
        if ti < len(df_clean):
            row = df_clean.iloc[ti]
            S_vals = [row[f'S_A{A}'] for A in A_values]
            ax1.loglog(A_values, S_vals, 'o-', color=colors[i],
                       label=f't={int(row["time"])}', alpha=0.8, linewidth=2)
    ax1.set_xlabel('Area (log scale)')
    ax1.set_ylabel('Species Richness (log scale)')
    ax1.set_title('SAR Power Law (Log-Log Plot)')
    ax1.legend()
    ax1.grid(True, which="both", ls="-", alpha=0.3)
    plt.tight_layout()
    # Save as PNG, PDF and SVG
    plt.savefig(os.path.join(output_dir, 'sar_loglog_multiple.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'sar_loglog_multiple.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'sar_loglog_multiple.svg'), bbox_inches='tight')
    plt.close(fig1)

    # Figure 2: Linear plot of SAR at different times
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    for i, ti in enumerate(sample_times):
        if ti < len(df_clean):
            row = df_clean.iloc[ti]
            S_vals = [row[f'S_A{A}'] for A in A_values]
            ax2.plot(A_values, S_vals, 'o-', color=colors[i],
                     label=f't={int(row["time"])}', alpha=0.8, linewidth=2)
    ax2.set_xlabel('Area')
    ax2.set_ylabel('Species Richness')
    ax2.set_title('SAR Curves at Different Times')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sar_linear_multiple.png'), dpi=300)
    plt.close(fig2)

    # Figure 3: Temporal dynamics of mean z-value
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    time_subset = df_clean['time'][::10]  # subsample to avoid crowding
    z_mean_subset = df_clean['z_mean'][::10]
    ax3.plot(time_subset, z_mean_subset, 'b-', alpha=0.7, linewidth=1.5)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Mean z-value')
    ax3.set_title('Temporal Dynamics of z-value')
    ax3.grid(True, alpha=0.3)
    z_mean_total = df_clean['z_mean'].mean()
    z_std_total = df_clean['z_mean'].std()
    ax3.axhline(y=z_mean_total, color='r', linestyle='--',
                label=f'Mean z = {z_mean_total:.3f} ± {z_std_total:.3f}')
    ax3.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'z_temporal_dynamics.png'), dpi=300)
    plt.close(fig3)

    # Figure 4: Distribution of SAR slopes (histogram)
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    slopes = df_clean['slope']
    ax4.hist(slopes, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax4.set_xlabel('Slope (z-value)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of SAR Slopes (z-values)')
    ax4.grid(True, alpha=0.3)
    slope_mean = slopes.mean()
    slope_std = slopes.std()
    ax4.axvline(slope_mean, color='red', linestyle='--',
                label=f'Mean = {slope_mean:.3f} ± {slope_std:.3f}')
    ax4.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'slope_distribution.png'), dpi=300)
    plt.close(fig4)

    # Figure 5: Single representative SAR power law with fit
    mid_time = len(df_clean) // 2
    row = df_clean.iloc[mid_time]
    S_vals = [row[f'S_A{A}'] for A in A_values]

    fig5, ax5 = plt.subplots(figsize=(10, 8))
    ax5.loglog(A_values, S_vals, 'bo-', linewidth=3, markersize=8, label='SAR Power Law')
    log_A = np.log(A_values)
    log_S = np.log([max(0.001, s) for s in S_vals])
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_A, log_S)
    fit_line = np.exp(intercept + slope * log_A)
    ax5.loglog(A_values, fit_line, 'r--', linewidth=2,
               label=f'Fit: S ∝ A^z\nz = {slope:.3f}, R² = {r_value**2:.3f}')
    ax5.set_xlabel('Area (log scale)', fontsize=12)
    ax5.set_ylabel('Species Richness (log scale)', fontsize=12)
    ax5.set_title('SAR Power Law with Fixed Extinction Baseline', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=12)
    ax5.grid(True, which="both", ls="-", alpha=0.3)
    ax5.text(0.02, 0.98,
             f'Model Parameters:\nInitial Species Pool = 1000\nD0(A) = 0\nE0(A) = 1 (Fixed)\nSp0 = 0',
             transform=ax5.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'perfect_sar_power_law.png'), dpi=300)
    plt.close(fig5)

# Call the plotting function
plot_sar_power_law(df_clean, output_dir)

# Output statistical results
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

# Additional verification: statistical characteristics of each SAR type
print("\n=== Statistical characteristics of each SAR type ===")
for sar_type in ['Positive SAR', 'Negative SAR', 'Non-SAR']:
    type_data = df_clean[df_clean['sar_type'] == sar_type]
    if len(type_data) > 0:
        print(f"\n{sar_type}:")
        print(f"  Sample count: {len(type_data)}")
        print(f"  z_mean range: [{type_data['z_mean'].min():.4f}, {type_data['z_mean'].max():.4f}]")
        print(f"  slope range: [{type_data['slope'].min():.4f}, {type_data['slope'].max():.4f}]")
        print(f"  R² range: [{type_data['r_squared'].min():.4f}, {type_data['r_squared'].max():.4f}]")

print("\nAnalysis complete! Modified results saved to", output_dir)