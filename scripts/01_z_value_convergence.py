import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import os
import itertools

class SAR2ConvergenceAnalyzer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Basic parameter settings
        self.c = 1.0  # constant
        self.A_values = [1, 5, 10, 20, 50, 100]  # area gradient
        self.t_max = 1000  # total time
        self.t = np.linspace(0, 1000, self.t_max)  # time axis
        
        # Process parameters - these will be varied in parameter scan
        self.ω_d, self.ω_e, self.ω_s = 0.1, 0.05, 0.01  # fluctuation frequencies
        self.τ_d = 0.5  # diffusion delay coefficient
        self.ϕ_e = 0.5  # extinction phase coefficient
        self.φ_s = 0.1  # speciation phase coefficient
        self.α_e = 0.8  # extinction amplitude
    
    def D(self, A, t, D0_A, ω_d, τ_d):
        """Diffusion process function"""
        return D0_A * (1 + np.sin(ω_d * t - τ_d * np.log(A)))
    
    def E(self, A, t, E0_A, ω_e, ϕ_e, α_e):
        """Extinction process function"""
        return E0_A * (1 + α_e * np.sin(ω_e * t + ϕ_e * np.log(A)))
    
    def Sp(self, A, t, Sp0_A, ω_s, φ_s):
        """Speciation process function"""
        return Sp0_A * (1 + np.sin(ω_s * t + φ_s * np.log(A)))
    
    def z(self, A, t, D0_A, E0_A, Sp0_A, ω_d, ω_e, ω_s, τ_d, ϕ_e, φ_s, α_e):
        """z-value calculation - includes all process parameters"""
        d_val = self.D(A, t, D0_A, ω_d, τ_d)
        e_val = self.E(A, t, E0_A, ω_e, ϕ_e, α_e)
        sp_val = self.Sp(A, t, Sp0_A, ω_s, φ_s)
        return d_val + sp_val - e_val
    
    def S(self, A, t, D0_A, E0_A, Sp0_A, ω_d, ω_e, ω_s, τ_d, ϕ_e, φ_s, α_e):
        """Species richness calculation"""
        z_val = self.z(A, t, D0_A, E0_A, Sp0_A, ω_d, ω_e, ω_s, τ_d, ϕ_e, φ_s, α_e)
        return self.c * (A ** z_val)
    
    def analyze_extended_parameter_space(self, 
                                       D0_A_range, E0_A_range, Sp0_A_range,
                                       ω_d_range, ω_e_range, ω_s_range,
                                       τ_d_range, ϕ_e_range, φ_s_range, α_e_range,
                                       n_samples=1000):
        """
        Analyze z-value convergence conditions in extended parameter space.
        
        Parameters:
        - ranges for each parameter
        - n_samples: number of samples
        """
        
        # Create parameter combination samples
        param_combinations = self._sample_parameter_combinations(
            D0_A_range, E0_A_range, Sp0_A_range,
            ω_d_range, ω_e_range, ω_s_range,
            τ_d_range, ϕ_e_range, φ_s_range, α_e_range,
            n_samples
        )
        
        results = []
        
        for i, params in enumerate(tqdm(param_combinations, desc="Scanning parameter space")):
            D0_A, E0_A, Sp0_A, ω_d, ω_e, ω_s, τ_d, ϕ_e, φ_s, α_e = params
            
            # Analyze z-value time series for each area
            z_time_series = {}
            z_spatial_variation = []
            
            for A in self.A_values:
                # Compute z-values for this area over time
                z_vals = [self.z(A, ti, D0_A, E0_A, Sp0_A, ω_d, ω_e, ω_s, τ_d, ϕ_e, φ_s, α_e) 
                         for ti in self.t[::10]]  # subsample time for speed
                z_time_series[A] = z_vals
                
                # Statistical characteristics of z-values
                z_mean = np.mean(z_vals)
                z_std = np.std(z_vals)
                z_spatial_variation.append(z_mean)
            
            # Evaluate convergence and SAR patterns
            convergence_metrics = self._evaluate_convergence_conditions_extended(
                z_time_series, z_spatial_variation
            )
            
            sar_metrics = self._calculate_sar_metrics_extended(
                z_time_series, D0_A, E0_A, Sp0_A, ω_d, ω_e, ω_s, τ_d, ϕ_e, φ_s, α_e
            )
            
            results.append({
                'D0_A': D0_A,
                'E0_A': E0_A,
                'Sp0_A': Sp0_A,
                'ω_d': ω_d,
                'ω_e': ω_e,
                'ω_s': ω_s,
                'τ_d': τ_d,
                'ϕ_e': ϕ_e,
                'φ_s': φ_s,
                'α_e': α_e,
                'mean_z': convergence_metrics['mean_z'],
                'z_time_std': convergence_metrics['z_time_std'],
                'z_space_std': convergence_metrics['z_space_std'],
                'classic_sar_prob': sar_metrics['classic_sar_prob'],
                'positive_sar_prob': sar_metrics['positive_sar_prob'],
                'mean_slope': sar_metrics['mean_slope'],
                'optimal_conditions': self._assess_optimal_conditions(convergence_metrics, sar_metrics)
            })
        
        return pd.DataFrame(results)
    
    def _sample_parameter_combinations(self, 
                                     D0_A_range, E0_A_range, Sp0_A_range,
                                     ω_d_range, ω_e_range, ω_s_range,
                                     τ_d_range, ϕ_e_range, φ_s_range, α_e_range,
                                     n_samples):
        """Generate parameter combinations using Latin Hypercube Sampling"""
        from scipy.stats import qmc
        
        # Define parameter bounds
        bounds = np.array([
            D0_A_range, E0_A_range, Sp0_A_range,
            ω_d_range, ω_e_range, ω_s_range,
            τ_d_range, ϕ_e_range, φ_s_range, α_e_range
        ])
        
        # Generate Latin Hypercube samples
        sampler = qmc.LatinHypercube(d=10)
        sample = sampler.random(n=n_samples)
        
        # Scale samples to parameter ranges
        param_samples = qmc.scale(sample, bounds[:, 0], bounds[:, 1])
        
        return param_samples
    
    def _evaluate_convergence_conditions_extended(self, z_time_series, z_spatial_variation):
        """Evaluate convergence conditions in extended parameter space"""
        
        metrics = {}
        
        # 1. Temporal stability: variation of z-values over time
        time_std_values = []
        for A, z_vals in z_time_series.items():
            # Standard deviation in later half of the time series
            half_idx = len(z_vals) // 2
            z_later = z_vals[half_idx:]
            time_std_values.append(np.std(z_later))
        
        metrics['z_time_std'] = np.mean(time_std_values)
        
        # 2. Spatial stability: variation among areas
        metrics['z_space_std'] = np.std(z_spatial_variation)
        
        # 3. Mean z-value
        metrics['mean_z'] = np.mean(z_spatial_variation)
        
        # 4. Proportion of positive z-values
        positive_z_ratio = []
        for A, z_vals in z_time_series.items():
            positive_count = np.sum(np.array(z_vals) > 0)
            positive_z_ratio.append(positive_count / len(z_vals))
        
        metrics['positive_z_ratio'] = np.mean(positive_z_ratio)
        
        return metrics
    
    def _calculate_sar_metrics_extended(self, z_time_series, D0_A, E0_A, Sp0_A, 
                                      ω_d, ω_e, ω_s, τ_d, ϕ_e, φ_s, α_e):
        """Calculate SAR metrics in extended parameter space"""
        
        classic_sar_detections = 0
        positive_sar_detections = 0
        slopes = []
        
        # Sample at multiple time points to test SAR relationships
        sample_times = self.t[::100]  # sample every 100 time points
        
        for ti in sample_times:
            # Compute species counts for each area at current time
            S_vals = [self.S(A, ti, D0_A, E0_A, Sp0_A, ω_d, ω_e, ω_s, τ_d, ϕ_e, φ_s, α_e) 
                     for A in self.A_values]
            
            # Test linear relationship between log(S) and log(A)
            log_A = np.log(self.A_values)
            log_S = np.log(S_vals)
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_A, log_S)
            slopes.append(slope)
            
            # Determine if classic SAR pattern (significant positive correlation)
            if p_value < 0.05 and slope > 0:
                positive_sar_detections += 1
                
                # Further check if classic SAR (z-value in typical range)
                if 0.1 < slope < 0.5:
                    classic_sar_detections += 1
        
        total_samples = len(sample_times)
        
        return {
            'classic_sar_prob': classic_sar_detections / total_samples if total_samples > 0 else 0,
            'positive_sar_prob': positive_sar_detections / total_samples if total_samples > 0 else 0,
            'mean_slope': np.mean(slopes) if slopes else 0
        }
    
    def _assess_optimal_conditions(self, convergence_metrics, sar_metrics):
        """Assess whether the parameter set yields optimal conditions (stable positive z and classic SAR)"""
        
        # Condition 1: z-value stability
        time_stable = convergence_metrics['z_time_std'] < 0.2
        space_stable = convergence_metrics['z_space_std'] < 0.3
        
        # Condition 2: positive z-values
        positive_z = convergence_metrics['mean_z'] > 0.1
        high_positive_ratio = convergence_metrics['positive_z_ratio'] > 0.7
        
        # Condition 3: classic SAR pattern
        high_sar_prob = sar_metrics['classic_sar_prob'] > 0.7
        positive_slope = sar_metrics['mean_slope'] > 0.1
        
        # Overall assessment
        return (time_stable and space_stable and 
                positive_z and high_positive_ratio and
                high_sar_prob and positive_slope)
    
    def plot_extended_analysis(self, df):
        """Generate separate figures for each subplot from the extended parameter space analysis."""
        
        # Filter parameter combinations that satisfy optimal conditions
        optimal_df = df[df['optimal_conditions']]
        
        print("=== Extended Parameter Space Analysis Results ===")
        print(f"Total parameter combinations: {len(df)}")
        print(f"Number meeting optimal conditions: {len(optimal_df)}")
        print(f"Optimal condition proportion: {len(optimal_df)/len(df):.3f}")
        
        if len(optimal_df) == 0:
            print("No optimal combinations found. Using top 10% by SAR probability for visualization.")
            # If none optimal, use top 10% by classic_sar_prob
            sar_threshold = df['classic_sar_prob'].quantile(0.9)
            optimal_df = df[df['classic_sar_prob'] >= sar_threshold]
        
        # Set style
        plt.style.use('default')
        
        # Parameters and their labels
        param_list = [
            ('D0_A', 'Diffusion Baseline'),
            ('E0_A', 'Extinction Baseline'),
            ('Sp0_A', 'Speciation Baseline'),
            ('ω_d', 'Diffusion Frequency'),
            ('ω_e', 'Extinction Frequency'),
            ('ω_s', 'Speciation Frequency'),
            ('τ_d', 'Diffusion Delay'),
            ('ϕ_e', 'Extinction Phase'),
            ('φ_s', 'Speciation Phase'),
            ('α_e', 'Extinction Amplitude')
        ]
        
        # -------------------------------
        # 1. Individual scatter plots for each parameter vs mean_z with SAR probability coloring
        # -------------------------------
        for i, (param, label) in enumerate(param_list):
            if param in optimal_df.columns:
                fig, ax = plt.subplots(figsize=(8, 6))
                scatter = ax.scatter(optimal_df[param], optimal_df['mean_z'], 
                                   c=optimal_df['classic_sar_prob'], cmap='viridis', alpha=0.7)
                ax.set_xlabel(label)
                ax.set_ylabel('Mean z-value')
                ax.set_title(f'{label} vs z-value\n(Color: SAR Probability)')
                ax.grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=ax, label='SAR Probability')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f'param_{param}_vs_z.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close(fig)
        
        # -------------------------------
        # 2. z-value distribution histogram
        # -------------------------------
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(optimal_df['mean_z'], bins=30, alpha=0.7, density=True)
        ax.set_xlabel('Mean z-value')
        ax.set_ylabel('Density')
        ax.set_title('z-value Distribution')
        ax.axvline(0.25, color='red', linestyle='--', label='Typical z-value (0.25)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        # Save as PNG (high resolution)
        plt.savefig(os.path.join(self.output_dir, 'z_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        # Save as PDF (vector)
        plt.savefig(os.path.join(self.output_dir, 'z_distribution.pdf'), 
                   bbox_inches='tight')
        # Save as SVG (vector)
        plt.savefig(os.path.join(self.output_dir, 'z_distribution.svg'), 
                   bbox_inches='tight')
        plt.close(fig)
        
        # -------------------------------
        # 3. SAR probability distribution histogram
        # -------------------------------
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(optimal_df['classic_sar_prob'], bins=30, alpha=0.7, density=True)
        ax.set_xlabel('Classic SAR Probability')
        ax.set_ylabel('Density')
        ax.set_title('SAR Probability Distribution')
        ax.axvline(0.7, color='red', linestyle='--', label='Threshold (0.7)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'sar_probability_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # -------------------------------
        # 4. Parameter correlation heatmap (already a separate function, but we call it here)
        # -------------------------------
        self._plot_parameter_correlations(optimal_df)
        
        # -------------------------------
        # 5. Parameter importance analysis (separate function)
        # -------------------------------
        self._plot_parameter_importance(optimal_df)
        
        # Output summary
        self._print_extended_summary(optimal_df, len(df))
        
        return optimal_df
    
    def _plot_parameter_correlations(self, df):
        """Plot parameter correlation heatmap"""
        if len(df) < 2:
            print("Insufficient data points for correlation calculation.")
            return
            
        # Select numeric columns for correlation
        numeric_cols = ['D0_A', 'E0_A', 'Sp0_A', 'ω_d', 'ω_e', 'ω_s', 
                       'τ_d', 'ϕ_e', 'φ_s', 'α_e', 'mean_z', 'classic_sar_prob']
        
        # Ensure all columns exist
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        if len(available_cols) < 3:
            print("Insufficient columns for correlation calculation.")
            return
            
        correlation_matrix = df[available_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(correlation_matrix.values, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, label='Correlation Coefficient')
        ax.set_xticks(range(len(available_cols)))
        ax.set_yticks(range(len(available_cols)))
        ax.set_xticklabels(available_cols, rotation=45)
        ax.set_yticklabels(available_cols)
        ax.set_title('Parameter Correlation Matrix')
        
        # Add correlation values
        for i in range(len(available_cols)):
            for j in range(len(available_cols)):
                ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', 
                        ha='center', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'parameter_correlations.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def _plot_parameter_importance(self, df):
        """Plot parameter importance analysis using Random Forest"""
        if len(df) < 2:
            print("Insufficient data points for importance calculation.")
            return
            
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score
        
        # Prepare features and target
        feature_cols = ['D0_A', 'E0_A', 'Sp0_A', 'ω_d', 'ω_e', 'ω_s', 
                       'τ_d', 'ϕ_e', 'φ_s', 'α_e']
        
        available_features = [col for col in feature_cols if col in df.columns]
        
        if len(available_features) < 2:
            print("Insufficient features for importance calculation.")
            return
            
        X = df[available_features]
        y = df['classic_sar_prob']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest model
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # Predict and compute R²
        y_pred = rf.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        # Get feature importance
        importance = rf.feature_importances_
        
        # Plot feature importance
        fig, ax = plt.subplots(figsize=(10, 6))
        indices = np.argsort(importance)[::-1]
        
        ax.bar(range(len(available_features)), importance[indices])
        ax.set_xticks(range(len(available_features)))
        ax.set_xticklabels([available_features[i] for i in indices], rotation=45)
        ax.set_xlabel('Parameters')
        ax.set_ylabel('Importance')
        ax.set_title(f'Parameter Importance for SAR Prediction (R² = {r2:.3f})')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'parameter_importance.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def _print_extended_summary(self, optimal_df, total_combinations):
        """Print summary of extended parameter space analysis"""
        
        print("\n=== Extended Parameter Space Analysis Summary ===")
        print(f"Total parameter combinations: {total_combinations}")
        print(f"Number meeting optimal conditions: {len(optimal_df)}")
        print(f"Optimal condition proportion: {len(optimal_df)/total_combinations:.3f}")
        
        if len(optimal_df) == 0:
            print("No parameter combinations meeting optimal conditions found.")
            print("Consider adjusting parameter ranges or convergence criteria.")
            return
        
        print("\nParameter ranges for optimal conditions (stable positive z and classic SAR):")
        
        # Baseline parameters
        print("\nBaseline parameters:")
        print(f"Diffusion baseline D0_A: [{optimal_df['D0_A'].min():.3f}, {optimal_df['D0_A'].max():.3f}]")
        print(f"Extinction baseline E0_A: [{optimal_df['E0_A'].min():.3f}, {optimal_df['E0_A'].max():.3f}]")
        print(f"Speciation baseline Sp0_A: [{optimal_df['Sp0_A'].min():.3f}, {optimal_df['Sp0_A'].max():.3f}]")
        
        # Frequency parameters
        print("\nFrequency parameters:")
        print(f"Diffusion frequency ω_d: [{optimal_df['ω_d'].min():.3f}, {optimal_df['ω_d'].max():.3f}]")
        print(f"Extinction frequency ω_e: [{optimal_df['ω_e'].min():.3f}, {optimal_df['ω_e'].max():.3f}]")
        print(f"Speciation frequency ω_s: [{optimal_df['ω_s'].min():.3f}, {optimal_df['ω_s'].max():.3f}]")
        
        # Phase and amplitude parameters
        print("\nPhase and amplitude parameters:")
        print(f"Diffusion delay τ_d: [{optimal_df['τ_d'].min():.3f}, {optimal_df['τ_d'].max():.3f}]")
        print(f"Extinction phase ϕ_e: [{optimal_df['ϕ_e'].min():.3f}, {optimal_df['ϕ_e'].max():.3f}]")
        print(f"Speciation phase φ_s: [{optimal_df['φ_s'].min():.3f}, {optimal_df['φ_s'].max():.3f}]")
        print(f"Extinction amplitude α_e: [{optimal_df['α_e'].min():.3f}, {optimal_df['α_e'].max():.3f}]")
        
        # Performance metrics
        print("\nPerformance metrics:")
        print(f"Mean z-value: [{optimal_df['mean_z'].min():.3f}, {optimal_df['mean_z'].max():.3f}]")
        print(f"Classic SAR probability: [{optimal_df['classic_sar_prob'].min():.3f}, {optimal_df['classic_sar_prob'].max():.3f}]")
        
        # Best parameter combination
        if 'classic_sar_prob' in optimal_df.columns and len(optimal_df) > 0:
            best_combination = optimal_df.loc[optimal_df['classic_sar_prob'].idxmax()]
            print(f"\nBest parameter combination (highest SAR probability):")
            print(f"D0_A={best_combination['D0_A']:.3f}, E0_A={best_combination['E0_A']:.3f}, Sp0_A={best_combination['Sp0_A']:.3f}")
            print(f"ω_d={best_combination['ω_d']:.3f}, ω_e={best_combination['ω_e']:.3f}, ω_s={best_combination['ω_s']:.3f}")
            print(f"τ_d={best_combination['τ_d']:.3f}, ϕ_e={best_combination['ϕ_e']:.3f}, φ_s={best_combination['φ_s']:.3f}, α_e={best_combination['α_e']:.3f}")
            print(f"Classic SAR probability: {best_combination['classic_sar_prob']:.3f}")
            print(f"Mean z-value: {best_combination['mean_z']:.3f}")
        
        # Save summary to file
        summary_file = os.path.join(self.output_dir, 'extended_analysis_summary.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=== Extended Parameter Space Analysis Summary ===\n")
            f.write(f"Total parameter combinations: {total_combinations}\n")
            f.write(f"Number meeting optimal conditions: {len(optimal_df)}\n")
            f.write(f"Optimal condition proportion: {len(optimal_df)/total_combinations:.3f}\n\n")
            
            if len(optimal_df) > 0:
                f.write("Parameter ranges for optimal conditions (stable positive z and classic SAR):\n\n")
                f.write("Baseline parameters:\n")
                f.write(f"Diffusion baseline D0_A: [{optimal_df['D0_A'].min():.3f}, {optimal_df['D0_A'].max():.3f}]\n")
                f.write(f"Extinction baseline E0_A: [{optimal_df['E0_A'].min():.3f}, {optimal_df['E0_A'].max():.3f}]\n")
                f.write(f"Speciation baseline Sp0_A: [{optimal_df['Sp0_A'].min():.3f}, {optimal_df['Sp0_A'].max():.3f}]\n\n")
                
                f.write("Frequency parameters:\n")
                f.write(f"Diffusion frequency ω_d: [{optimal_df['ω_d'].min():.3f}, {optimal_df['ω_d'].max():.3f}]\n")
                f.write(f"Extinction frequency ω_e: [{optimal_df['ω_e'].min():.3f}, {optimal_df['ω_e'].max():.3f}]\n")
                f.write(f"Speciation frequency ω_s: [{optimal_df['ω_s'].min():.3f}, {optimal_df['ω_s'].max():.3f}]\n\n")
                
                f.write("Phase and amplitude parameters:\n")
                f.write(f"Diffusion delay τ_d: [{optimal_df['τ_d'].min():.3f}, {optimal_df['τ_d'].max():.3f}]\n")
                f.write(f"Extinction phase ϕ_e: [{optimal_df['ϕ_e'].min():.3f}, {optimal_df['ϕ_e'].max():.3f}]\n")
                f.write(f"Speciation phase φ_s: [{optimal_df['φ_s'].min():.3f}, {optimal_df['φ_s'].max():.3f}]\n")
                f.write(f"Extinction amplitude α_e: [{optimal_df['α_e'].min():.3f}, {optimal_df['α_e'].max():.3f}]\n\n")
                
                f.write("Performance metrics:\n")
                f.write(f"Mean z-value: [{optimal_df['mean_z'].min():.3f}, {optimal_df['mean_z'].max():.3f}]\n")
                f.write(f"Classic SAR probability: [{optimal_df['classic_sar_prob'].min():.3f}, {optimal_df['classic_sar_prob'].max():.3f}]\n\n")
                
                if 'classic_sar_prob' in optimal_df.columns:
                    best_combination = optimal_df.loc[optimal_df['classic_sar_prob'].idxmax()]
                    f.write("Best parameter combination (highest SAR probability):\n")
                    f.write(f"D0_A={best_combination['D0_A']:.3f}, E0_A={best_combination['E0_A']:.3f}, Sp0_A={best_combination['Sp0_A']:.3f}\n")
                    f.write(f"ω_d={best_combination['ω_d']:.3f}, ω_e={best_combination['ω_e']:.3f}, ω_s={best_combination['ω_s']:.3f}\n")
                    f.write(f"τ_d={best_combination['τ_d']:.3f}, ϕ_e={best_combination['ϕ_e']:.3f}, φ_s={best_combination['φ_s']:.3f}, α_e={best_combination['α_e']:.3f}\n")
                    f.write(f"Classic SAR probability: {best_combination['classic_sar_prob']:.3f}\n")
                    f.write(f"Mean z-value: {best_combination['mean_z']:.3f}\n")
            else:
                f.write("No parameter combinations meeting optimal conditions found.\n")
                f.write("Consider adjusting parameter ranges or convergence criteria.\n")
    
    def save_extended_results(self, df, optimal_df):
        """Save extended analysis results to CSV files"""
        df.to_csv(os.path.join(self.output_dir, 'extended_parameter_analysis.csv'), index=False)
        
        if optimal_df is not None and len(optimal_df) > 0:
            optimal_df.to_csv(os.path.join(self.output_dir, 'optimal_parameters_extended.csv'), index=False)
        
        print(f"Extended analysis results saved to {self.output_dir}")

def main():
    """Main analysis function"""
    output_dir = os.path.join(os.getcwd(), "results")
    analyzer = SAR2ConvergenceAnalyzer(output_dir)
    
    # Define scanning ranges for extended parameter space
    # Baseline parameters
    D0_A_range = [0.01, 1.0]  # diffusion baseline
    E0_A_range = [0.01, 1.0]  # extinction baseline
    Sp0_A_range = [0.01, 0.1]  # speciation baseline
    
    # Frequency parameters
    ω_d_range = [0.01, 0.5]   # diffusion frequency
    ω_e_range = [0.01, 0.1]   # extinction frequency
    ω_s_range = [0.001, 0.01] # speciation frequency
    
    # Phase and amplitude parameters
    τ_d_range = [0.1, 1.0]    # diffusion delay
    ϕ_e_range = [0.1, 1.0]    # extinction phase
    φ_s_range = [0.05, 0.2]   # speciation phase
    α_e_range = [0.5, 1.0]    # extinction amplitude
    
    print("Starting extended parameter space scan...")
    results_df = analyzer.analyze_extended_parameter_space(
        D0_A_range, E0_A_range, Sp0_A_range,
        ω_d_range, ω_e_range, ω_s_range,
        τ_d_range, ϕ_e_range, φ_s_range, α_e_range,
        n_samples=100000  # reduced for speed
    )
    
    print("Generating extended parameter space analysis figures...")
    optimal_df = analyzer.plot_extended_analysis(results_df)
    
    # Save results to file
    analyzer.save_extended_results(results_df, optimal_df)
    
    print(f"Extended parameter space analysis complete! Results saved to {output_dir}")
    return analyzer, results_df, optimal_df

if __name__ == "__main__":
    analyzer, results_df, optimal_df = main()