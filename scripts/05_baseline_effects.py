import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import os
from scipy import stats

# Set uniform font and image quality
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300

class SAR2_Model:
    def __init__(self, time_steps=1000, num_samples=200):
        """
        SAR2.0 model initialization - extinction-dominated ecological model
        time_steps: number of time steps
        num_samples: number of sampling points - increased to 200 to obtain more sampling rectangles
        """
        self.time_steps = time_steps
        self.num_samples = num_samples
        self.time_points = np.linspace(0, time_steps, num_samples)
        
        # SAR2.0 model parameters
        self.c = 10000.0  # increase constant c to provide larger baseline species pool
        self.A_values = [1, 5, 10, 20, 50, 100]  # area gradient
        
        # Process parameters - extinction-dominated
        self.D0_A = 0.5  # reduce diffusion baseline intensity
        self.E0_A = 1.0  # increase extinction baseline intensity to make it dominant
        self.Sp0_A = 0   # reduce speciation baseline intensity
        
        # Fluctuation parameters - enhance extinction fluctuations
        self.ω_d, self.ω_e, self.ω_s = 0.1, 0.05, 0.01  # adjust fluctuation frequencies to enhance extinction fluctuations
        self.τ_d = 0.5   # diffusion delay coefficient
        self.ϕ_e = 0.5   # extinction phase coefficient
        self.φ_s = 0.1   # speciation phase coefficient
        self.α_e = 2     # increase extinction fluctuation amplitude
        
        # Noise parameters - newly added
        self.noise_level = 0.15  # increase noise level to make deviations more pronounced
        self.random_state = np.random.RandomState(42)  # fix random seed for reproducibility
    
    def D(self, A, t):
        """Diffusion process function - controlled, with added noise"""
        base = self.D0_A * (1 + 0.3 * np.sin(self.ω_d * t - self.τ_d * np.log(A)))
        # Add noise
        noise = self.random_state.normal(0, self.noise_level * 0.5)  # smaller noise for diffusion process
        return max(0, base + noise)  # ensure non-negative
    
    def E(self, A, t):
        """Extinction process function - dominant with fluctuations, added noise"""
        base = self.E0_A * (1 + self.α_e * np.sin(self.ω_e * t + self.ϕ_e * np.log(A)))
        # Add noise
        noise = self.random_state.normal(0, self.noise_level)
        return max(0, base + noise)  # ensure non-negative
    
    def Sp(self, A, t):
        """Speciation process function - controlled, added noise"""
        base = self.Sp0_A * (1 + 0.2 * np.sin(self.ω_s * t + self.φ_s * np.log(A)))
        # Add noise
        noise = self.random_state.normal(0, self.noise_level * 0.3)  # smaller noise for speciation process
        return max(0, base + noise)  # ensure non-negative
    
    def z(self, A, t):
        """z(A,t) function"""
        return self.D(A, t) + self.Sp(A, t) - self.E(A, t)
    
    def S(self, A, t):
        """Species richness function"""
        z_val = self.z(A, t)
        # Add some noise to species richness calculation
        noise = self.random_state.normal(1, 0.05)  # 5% noise
        return self.c * (A ** z_val) * noise
    
    def calculate_sar_correlation(self, t):
        """
        Calculate SAR correlation
        Obtain SAR slope by linear regression of species richness against area
        """
        # Compute species richness for each area at current time
        S_values = [self.S(A, t) for A in self.A_values]
        
        # Perform linear regression on log(S) ~ log(A)
        log_A = np.log(self.A_values)
        log_S = np.log(S_values)
        
        # Compute correlation coefficients
        if len(log_A) > 1 and len(log_S) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_A, log_S)
            return slope, r_value, p_value
        else:
            return 0, 0, 1
    
    def generate_sar_data(self):
        """Generate SAR data"""
        data = []
        
        for i, t in enumerate(self.time_points):
            # Compute SAR correlation
            slope, r_value, p_value = self.calculate_sar_correlation(t)
            
            # Compute mean of z-values
            z_vals = [self.z(A, t) for A in self.A_values]
            avg_z = np.mean(z_vals)
            
            # Add random noise to z-values so they are not perfectly sinusoidal
            # Noise magnitude is proportional to current z-value, but does not exceed set noise level
            base_noise = self.random_state.normal(0, self.noise_level * abs(avg_z) if abs(avg_z) > 0 else self.noise_level * 0.1)
            
            # Add extra time‑dependent noise to create more regular deviations
            time_noise = 0.05 * np.sin(0.3 * t) * self.random_state.normal(0, 0.5)
            
            noisy_z = avg_z + base_noise + time_noise
            
            # Determine SAR pattern and corresponding z-value based on slope and p-value
            # Use noisy z-values for classification
            if slope > 0.01 and p_value < 0.05:  # significant positive correlation
                phase = 'positive'
                forest_z = noisy_z  # use noisy z-values
            elif slope < -0.01 and p_value < 0.05:  # significant negative correlation
                phase = 'negative'
                forest_z = noisy_z  # use noisy z-values
            else:  # no significant correlation
                phase = 'neutral'
                # For neutral phases, add some randomness so some points deviate slightly from zero line
                if self.random_state.random() < 0.4:  # 40% chance of small deviation
                    small_noise = self.random_state.normal(0, 0.03)
                    forest_z = small_noise
                else:
                    forest_z = 0  # treat z-value as 0
            
            data.append({
                'time': t,
                'sar_slope': slope,
                'r_value': r_value,
                'p_value': p_value,
                'phase': phase,
                'z_value': avg_z,  # actual z-value (without noise)
                'noisy_z': noisy_z,  # z-value with noise
                'forest_z': forest_z,  # z-value used in forest plot (with noise)
                'sample_id': i
            })
        
        return pd.DataFrame(data)
    
    def plot_sar_forest(self, df, save_path=None):
        """Plot SAR forest plot – using uniform visual style"""
        # Use uniform figure size: 18×8 inches
        fig, ax = plt.subplots(figsize=(18, 8))
        
        # Compute time interval to avoid overlapping rectangles
        time_interval = self.time_steps / self.num_samples
        # Rectangle width
        block_width = time_interval * 0.7
        
        # Define uniform color scheme
        colors = {
            'positive': '#E74C3C',  # red – positive correlation
            'negative': '#3498DB',  # blue – negative correlation
            'neutral': '#95A5A6'    # grey – no correlation
        }
        
        # Main plot – forest plot
        for _, row in df.iterrows():
            time_val = row['time']
            forest_z = row['forest_z']
            
            # Determine rectangle color according to SAR type
            color = colors[row['phase']]
            
            # Draw rectangle – using forest_z as y-coordinate
            rect = patches.Rectangle((time_val - block_width/2, forest_z - 0.025), 
                                   block_width, 0.05,
                                   linewidth=0.8, edgecolor='black',
                                   facecolor=color, alpha=0.7)
            ax.add_patch(rect)
        
        # Set main plot properties – horizontal axis line
        ax.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.7)
        
        # Use uniform font settings
        ax.set_xlabel('Time', fontsize=14, fontweight='bold')
        ax.set_ylabel('Z Value', fontsize=14, fontweight='bold')
        ax.set_title('Extinction-Dominated SAR Discontinuity Forest Plot', 
                    fontsize=16, fontweight='bold')
        
        # Axis tick labels
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Background grid – uniform alpha value
        ax.grid(True, alpha=0.3)
        
        # Set y‑axis range
        z_min = min(df['forest_z'].min(), -0.15)
        z_max = max(df['forest_z'].max(), 0.15)
        ax.set_ylim(z_min - 0.1, z_max + 0.1)
        
        # x‑axis range: 0–200
        ax.set_xlim(0, 200)
        
        # Add legend – uniform legend style
        legend_elements = [
            patches.Patch(facecolor=colors['positive'], alpha=0.7, edgecolor='black', 
                         label='Positive Correlation'),
            patches.Patch(facecolor=colors['neutral'], alpha=0.7, edgecolor='black', 
                         label='No Correlation'),
            patches.Patch(facecolor=colors['negative'], alpha=0.7, edgecolor='black', 
                         label='Negative Correlation')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12, 
                 frameon=True, fancybox=True)
        
        plt.tight_layout()
        
        # Save image in multiple formats
        if save_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save as PNG (original path)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Forest plot saved (PNG): {save_path}")
            
            # Derive PDF and SVG paths by replacing extension
            base, ext = os.path.splitext(save_path)
            pdf_path = base + '.pdf'
            svg_path = base + '.svg'
            
            plt.savefig(pdf_path, bbox_inches='tight')
            print(f"Forest plot saved (PDF): {pdf_path}")
            
            plt.savefig(svg_path, bbox_inches='tight')
            print(f"Forest plot saved (SVG): {svg_path}")
        
        plt.show()
        
        return fig

    def calculate_statistics(self, df):
        """Compute statistical information"""
        stats = {
            'total_samples': len(df),
            'positive_correlation': len(df[df['phase'] == 'positive']),
            'negative_correlation': len(df[df['phase'] == 'negative']),
            'neutral_correlation': len(df[df['phase'] == 'neutral']),
        }
        
        print("\n=== SAR2.0 Model Statistics: Extinction-Dominated Habitat ===")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Positive correlation: {stats['positive_correlation']} ({stats['positive_correlation']/stats['total_samples']*100:.1f}%)")
        print(f"Negative correlation: {stats['negative_correlation']} ({stats['negative_correlation']/stats['total_samples']*100:.1f}%)")
        print(f"Neutral correlation: {stats['neutral_correlation']} ({stats['neutral_correlation']/stats['total_samples']*100:.1f}%)")
        
        return stats

def main():
    """Main function"""
    # Initialize SAR2.0 model – extinction-dominated habitat
    sar_model = SAR2_Model(time_steps=1000, num_samples=200)
    
    # Generate SAR data
    print("Generating SAR2.0 model data for extinction-dominated habitat...")
    sar_data = sar_model.generate_sar_data()
    
    # Compute statistics
    statistics = sar_model.calculate_statistics(sar_data)
    
    # Set save path
    save_directory = os.path.join(os.getcwd(), "results")
    
    # Plot forest plot
    forest_plot_path = os.path.join(save_directory, "extinction_SAR_forest_plot.png")
    print("Generating forest plot...")
    forest_fig = sar_model.plot_sar_forest(sar_data, save_path=forest_plot_path)

def detect_discontinuities(df, threshold=0.1):
    """Detect temporal discontinuity points"""
    print("\n=== Temporal Discontinuity Analysis ===")
    discontinuities = []
    
    for i in range(1, len(df)):
        # Check phase change
        if df.iloc[i]['phase'] != df.iloc[i-1]['phase']:
            discontinuities.append({
                'time': df.iloc[i]['time'],
                'from_phase': df.iloc[i-1]['phase'],
                'to_phase': df.iloc[i]['phase'],
            })
    
    if discontinuities:
        print(f"Detected {len(discontinuities)} phase transition points:")
        for disc in discontinuities:
            print(f"  Time {disc['time']:.1f}: {disc['from_phase']} → {disc['to_phase']}")
    else:
        print("No significant temporal discontinuities detected")
    
    return discontinuities

if __name__ == "__main__":
    main()