import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy import stats
from tqdm import tqdm

# Set uniform font and image quality
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300

# Create output directory
output_dir = os.path.join(os.getcwd(), "results")
os.makedirs(output_dir, exist_ok=True)

# Parameter settings
c = 1.0  # constant
A_values = [1, 5, 10, 20, 50, 100]  # area gradient
t_max = 1000  # total time
t = np.linspace(0, 1000, t_max)  # time axis

# Other process parameters
ω_d, ω_e, ω_s = 0.1, 0.05, 0.01  # fluctuation frequencies
τ_d = 0.5   # diffusion delay coefficient
ϕ_e = 0.5   # extinction phase coefficient
φ_s = 0.1   # speciation phase coefficient
α_e = 0.8   # extinction amplitude

# Define the three process functions
def D(A, t, D0_A):
    return D0_A * (1 + np.sin(ω_d * t - τ_d * np.log(A)))

def E(A, t, E0_A):
    return E0_A * (1 + α_e * np.sin(ω_e * t + ϕ_e * np.log(A)))

def Sp(A, t, Sp0_A):
    return Sp0_A * (1 + np.sin(ω_s * t + φ_s * np.log(A)))

# Define z(A,t) function – exponent in SAR formula
def z(A, t, D0_A, E0_A, Sp0_A):
    return D(A, t, D0_A) + Sp(A, t, Sp0_A) - E(A, t, E0_A)

# Define species richness function
def S(A, t, D0_A, E0_A, Sp0_A, c=1.0):
    z_val = z(A, t, D0_A, E0_A, Sp0_A)
    return c * (A ** z_val)

# Simulation function: compute SAR proportions for given baseline values
def simulate_sar_ratios(D0_A_val, E0_A_val, Sp0_A_val):
    # Use all time points as samples
    sample_times = t
    
    results = []
    
    for i, ti in enumerate(sample_times):
        # Compute species counts for each area at current time
        S_vals = [S(A, ti, D0_A_val, E0_A_val, Sp0_A_val) for A in A_values]
        
        # Compute significance of SAR pattern (p-value)
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
        
        results.append({
            'time': ti,
            'sar_type': sar_type,
            'slope': slope,
            'p_value': p_value,
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Extract three SAR types
    positive_sar = df[df['sar_type'] == "Positive SAR"]
    negative_sar = df[df['sar_type'] == "Negative SAR"]
    non_sar = df[df['sar_type'] == "Non-SAR"]
    
    # Compute proportions of the three SAR types
    total_samples = len(df)
    positive_ratio = len(positive_sar) / total_samples
    negative_ratio = len(negative_sar) / total_samples
    non_ratio = len(non_sar) / total_samples
    
    return positive_ratio, negative_ratio, non_ratio

# Define baseline ranges – 1000 steps
D0_A_range = np.linspace(0.01, 1.0, 1000)  # diffusion baseline range
E0_A_range = np.linspace(0.01, 1.0, 1000)  # extinction baseline range
Sp0_A_range = np.linspace(0.01, 0.1, 1000) # speciation baseline range

# Fix two parameters, vary one, and analyze its impact on SAR proportions
results_D = []
results_E = []
results_Sp = []

# Fix E0_A and Sp0_A, vary D0_A
fixed_E0_A = 0.1
fixed_Sp0_A = 0.01
print("Analyzing impact of diffusion baseline on SAR proportions...")
for D0_A_val in tqdm(D0_A_range):
    pos_ratio, neg_ratio, non_ratio = simulate_sar_ratios(D0_A_val, fixed_E0_A, fixed_Sp0_A)
    results_D.append({
        'D0_A': D0_A_val,
        'Positive_SAR': pos_ratio,
        'Negative_SAR': neg_ratio,
        'Non_SAR': non_ratio
    })

# Fix D0_A and Sp0_A, vary E0_A
fixed_D0_A = 1
fixed_Sp0_A = 0.01
print("Analyzing impact of extinction baseline on SAR proportions...")
for E0_A_val in tqdm(E0_A_range):
    pos_ratio, neg_ratio, non_ratio = simulate_sar_ratios(fixed_D0_A, E0_A_val, fixed_Sp0_A)
    results_E.append({
        'E0_A': E0_A_val,
        'Positive_SAR': pos_ratio,
        'Negative_SAR': neg_ratio,
        'Non_SAR': non_ratio
    })

# Fix D0_A and E0_A, vary Sp0_A
fixed_D0_A = 0.1
fixed_E0_A = 0.05
print("Analyzing impact of speciation baseline on SAR proportions...")
for Sp0_A_val in tqdm(Sp0_A_range):
    pos_ratio, neg_ratio, non_ratio = simulate_sar_ratios(fixed_D0_A, fixed_E0_A, Sp0_A_val)
    results_Sp.append({
        'Sp0_A': Sp0_A_val,
        'Positive_SAR': pos_ratio,
        'Negative_SAR': neg_ratio,
        'Non_SAR': non_ratio
    })

# Convert to DataFrames
df_D = pd.DataFrame(results_D)
df_E = pd.DataFrame(results_E)
df_Sp = pd.DataFrame(results_Sp)

# Save results
df_D.to_csv(os.path.join(output_dir, 'baseline_D_impact.csv'), index=False)
df_E.to_csv(os.path.join(output_dir, 'baseline_E_impact.csv'), index=False)
df_Sp.to_csv(os.path.join(output_dir, 'baseline_Sp_impact.csv'), index=False)

# -------------------------------------------------------------------
# Define uniform color scheme
colors = {
    'positive': '#E74C3C',  # red – positive correlation
    'negative': '#3498DB',  # blue – negative correlation
    'neutral': '#2ECC71'    # green – no correlation
}

# -------------------------------------------------------------------
# Figure 1: Impact of diffusion baseline on SAR proportions
# -------------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(6, 6))
ax1.plot(df_D['D0_A'], df_D['Positive_SAR'], color=colors['positive'], 
         linewidth=2.5, label='Positive SAR')
ax1.plot(df_D['D0_A'], df_D['Negative_SAR'], color=colors['negative'], 
         linewidth=2.5, label='Negative SAR')
ax1.plot(df_D['D0_A'], df_D['Non_SAR'], color=colors['neutral'], 
         linewidth=2.5, label='No Correlation')

ax1.set_xlabel('Diffusion Baseline (D0)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Proportion', fontsize=12, fontweight='bold')
ax1.set_title('Diffusion Baseline Impact\n(E0=0.1, Sp0=0.01)', 
              fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, frameon=True, fancybox=True)
ax1.set_xlim(0.1, 1.0)
ax1.set_ylim(0, 1)
ax1.grid(True, alpha=0.3)
plt.tight_layout()

# Save as PNG, PDF and SVG
base_path = os.path.join(output_dir, 'diffusion_baseline_impact')
plt.savefig(base_path + '.png', dpi=300, bbox_inches='tight')
plt.savefig(base_path + '.pdf', bbox_inches='tight')
plt.savefig(base_path + '.svg', bbox_inches='tight')
plt.close(fig1)

# -------------------------------------------------------------------
# Figure 2: Impact of extinction baseline on SAR proportions
# -------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(6, 6))
ax2.plot(df_E['E0_A'], df_E['Positive_SAR'], color=colors['positive'], 
         linewidth=2.5, label='Positive SAR')
ax2.plot(df_E['E0_A'], df_E['Negative_SAR'], color=colors['negative'], 
         linewidth=2.5, label='Negative SAR')
ax2.plot(df_E['E0_A'], df_E['Non_SAR'], color=colors['neutral'], 
         linewidth=2.5, label='No Correlation')

ax2.set_xlabel('Extinction Baseline (E0)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Proportion', fontsize=12, fontweight='bold')
ax2.set_title('Extinction Baseline Impact\n(D0=1.0, Sp0=0.01)', 
              fontsize=14, fontweight='bold')
ax2.legend(fontsize=10, frameon=True, fancybox=True)
ax2.set_xlim(0.1, 1.0)
ax2.set_ylim(0, 1)
ax2.grid(True, alpha=0.3)
plt.tight_layout()

base_path = os.path.join(output_dir, 'extinction_baseline_impact')
plt.savefig(base_path + '.png', dpi=300, bbox_inches='tight')
plt.savefig(base_path + '.pdf', bbox_inches='tight')
plt.savefig(base_path + '.svg', bbox_inches='tight')
plt.close(fig2)

# -------------------------------------------------------------------
# Figure 3: Impact of speciation baseline on SAR proportions
# -------------------------------------------------------------------
fig3, ax3 = plt.subplots(figsize=(6, 6))
ax3.plot(df_Sp['Sp0_A'], df_Sp['Positive_SAR'], color=colors['positive'], 
         linewidth=2.5, label='Positive SAR')
ax3.plot(df_Sp['Sp0_A'], df_Sp['Negative_SAR'], color=colors['negative'], 
         linewidth=2.5, label='Negative SAR')
ax3.plot(df_Sp['Sp0_A'], df_Sp['Non_SAR'], color=colors['neutral'], 
         linewidth=2.5, label='No Correlation')

ax3.set_xlabel('Speciation Baseline (Sp0)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Proportion', fontsize=12, fontweight='bold')
ax3.set_title('Speciation Baseline Impact\n(D0=0.1, E0=0.05)', 
              fontsize=14, fontweight='bold')
ax3.legend(fontsize=10, frameon=True, fancybox=True)
ax3.set_xlim(0.01, 0.1)
ax3.set_ylim(0, 1)
ax3.grid(True, alpha=0.3)
plt.tight_layout()

base_path = os.path.join(output_dir, 'speciation_baseline_impact')
plt.savefig(base_path + '.png', dpi=300, bbox_inches='tight')
plt.savefig(base_path + '.pdf', bbox_inches='tight')
plt.savefig(base_path + '.svg', bbox_inches='tight')
plt.close(fig3)

print("Analysis complete! Results saved to", output_dir)