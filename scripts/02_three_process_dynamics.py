import numpy as np
import matplotlib.pyplot as plt
import os

# Set publication-quality style
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300

# Output directory
output_dir = os.path.join(os.getcwd(), "results")
os.makedirs(output_dir, exist_ok=True)

# Model parameters
c = 1.0
A_values = [1, 5, 10, 20, 50, 100]
t_max = 500
t = np.linspace(0, 500, t_max)

D0_A = 1.00
E0_A = 0.80
Sp0_A = 0.01

ω_d, ω_e, ω_s = 0.10, 0.05, 0.01
τ_d = 0.50
ϕ_e = 0.50
φ_s = 0.10
α_e = 0.80

# Process functions
def D(A, t):
    return D0_A * (1 + np.sin(ω_d * t - τ_d * np.log(A)))

def E(A, t):
    return E0_A * (1 + α_e * np.sin(ω_e * t + ϕ_e * np.log(A)))

def Sp(A, t):
    return Sp0_A * (1 + np.sin(ω_s * t + φ_s * np.log(A)))

def z(A, t):
    return D(A, t) + Sp(A, t) - E(A, t)

# Colors and line styles (consistent across figures)
colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C']
line_styles = ['-', '--', '-.', ':', '-', '--']

# ------------------------------------------------------------
# Figure A: Diffusion Process Dynamics (title without (A))
# ------------------------------------------------------------
fig_a, ax_a = plt.subplots(figsize=(10, 6))
for i, A in enumerate(A_values):
    D_vals = [D(A, ti) for ti in t]
    ax_a.plot(t, D_vals, color=colors[i], linestyle=line_styles[i],
             linewidth=2, label=f'A={A}')
ax_a.set_xlabel('Time', fontsize=12, fontweight='bold')
ax_a.set_ylabel('Diffusion Intensity', fontsize=12, fontweight='bold')
ax_a.set_title('Diffusion Process Dynamics', fontsize=14, fontweight='bold')
ax_a.legend(fontsize=9, frameon=True, fancybox=True)
ax_a.grid(True, alpha=0.3)
ax_a.set_xlim(0, 200)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figure_A_diffusion.png'), bbox_inches='tight', dpi=300)
plt.close(fig_a)

# ------------------------------------------------------------
# Figure B: Extinction Process Dynamics (title without (B))
# ------------------------------------------------------------
fig_b, ax_b = plt.subplots(figsize=(10, 6))
for i, A in enumerate(A_values):
    E_vals = [E(A, ti) for ti in t]
    ax_b.plot(t, E_vals, color=colors[i], linestyle=line_styles[i],
             linewidth=2, label=f'A={A}')
ax_b.set_xlabel('Time', fontsize=12, fontweight='bold')
ax_b.set_ylabel('Extinction Intensity', fontsize=12, fontweight='bold')
ax_b.set_title('Extinction Process Dynamics', fontsize=14, fontweight='bold')
ax_b.legend(fontsize=9, frameon=True, fancybox=True)
ax_b.grid(True, alpha=0.3)
ax_b.set_xlim(0, 200)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figure_B_extinction.png'), bbox_inches='tight', dpi=300)
plt.close(fig_b)

# ------------------------------------------------------------
# Figure C: Speciation Process Dynamics (title without (C))
# ------------------------------------------------------------
fig_c, ax_c = plt.subplots(figsize=(10, 6))
for i, A in enumerate(A_values):
    Sp_vals = [Sp(A, ti) for ti in t]
    ax_c.plot(t, Sp_vals, color=colors[i], linestyle=line_styles[i],
             linewidth=2, label=f'A={A}')
ax_c.set_xlabel('Time', fontsize=12, fontweight='bold')
ax_c.set_ylabel('Speciation Intensity', fontsize=12, fontweight='bold')
ax_c.set_title('Speciation Process Dynamics', fontsize=14, fontweight='bold')
ax_c.legend(fontsize=9, frameon=True, fancybox=True)
ax_c.grid(True, alpha=0.3)
ax_c.set_xlim(0, 200)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figure_C_speciation.png'), bbox_inches='tight', dpi=300)
plt.close(fig_c)

# ------------------------------------------------------------
# Figure F: Net Z Value Fluctuation (title without (F), saved as PNG+PDF+SVG)
# ------------------------------------------------------------
fig_f, ax_f = plt.subplots(figsize=(10, 6))
for i, A in enumerate(A_values):
    z_vals_net = [z(A, ti) for ti in t]
    ax_f.plot(t, z_vals_net, color=colors[i], linestyle=line_styles[i],
             linewidth=2, label=f'A={A}')
ax_f.set_xlabel('Time', fontsize=12, fontweight='bold')
ax_f.set_ylabel('Net Z Value (D - E + Sp)', fontsize=12, fontweight='bold')
ax_f.set_title('Net Z Value Fluctuation', fontsize=14, fontweight='bold')
ax_f.legend(fontsize=9, frameon=True, fancybox=True)
ax_f.grid(True, alpha=0.3)
ax_f.set_xlim(0, 200)
plt.tight_layout()

# Save as PNG (renamed to figure_netZ.png)
plt.savefig(os.path.join(output_dir, 'figure_netZ.png'), bbox_inches='tight', dpi=300)
# Save as PDF (vector)
plt.savefig(os.path.join(output_dir, 'figure_netZ.pdf'), bbox_inches='tight')
# Save as SVG (vector)
plt.savefig(os.path.join(output_dir, 'figure_netZ.svg'), bbox_inches='tight')
plt.close(fig_f)

print("Four figures generated successfully:")
print("1. figure_A_diffusion.png")
print("2. figure_B_extinction.png")
print("3. figure_C_speciation.png")
print("4. figure_netZ.png (also saved as PDF and SVG)")
print(f"All saved to: {output_dir}")