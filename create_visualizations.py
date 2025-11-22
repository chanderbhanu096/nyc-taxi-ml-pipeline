import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style for professional look
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# Model performance data
models = ['Random Forest', 'Gradient Boosting', 'Decision Tree', 'XGBoost', 
          'KNN', 'ElasticNet', 'Lasso', 'Ridge', 'Linear']
mae_scores = [1.05, 1.07, 1.08, 1.13, 1.16, 4.58, 4.58, 4.59, 4.59]
r2_scores = [0.9475, 0.9457, 0.9445, 0.9196, 0.9435, 0.6669, 0.6663, 0.6688, 0.6688]

# Create figure with better styling
fig = plt.figure(figsize=(18, 7))
gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.25)

# Define modern color palette
colors_mae = ['#2ecc71', '#27ae60', '#16a085', '#3498db', '#2980b9', 
              '#e74c3c', '#c0392b', '#e67e22', '#d35400']

# Plot 1: MAE Comparison with gradient colors
ax1 = fig.add_subplot(gs[0, 0])
bars1 = ax1.barh(range(len(models)), mae_scores, color=colors_mae, 
                 edgecolor='white', linewidth=2, alpha=0.9)

# Add gradient effect
for i, (bar, val) in enumerate(zip(bars1, mae_scores)):
    bar.set_height(0.7)
    # Add value labels
    ax1.text(val + 0.15, i, f'${val:.2f}', 
             va='center', ha='left', fontsize=11, fontweight='bold')

ax1.set_yticks(range(len(models)))
ax1.set_yticklabels(models, fontsize=11)
ax1.set_xlabel('Mean Absolute Error ($)', fontsize=13, fontweight='bold')
ax1.set_title('Model Performance: MAE Comparison\n(Lower is Better)', 
              fontsize=15, fontweight='bold', pad=20)
ax1.axvline(x=2, color='gray', linestyle='--', linewidth=1.5, alpha=0.4, label='Threshold')
ax1.grid(axis='x', alpha=0.2, linestyle='-', linewidth=0.5)
ax1.set_xlim(0, max(mae_scores) + 1)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Plot 2: R² Score Comparison
ax2 = fig.add_subplot(gs[0, 1])
bars2 = ax2.barh(range(len(models)), r2_scores, color=colors_mae,
                 edgecolor='white', linewidth=2, alpha=0.9)

# Add gradient effect
for i, (bar, val) in enumerate(zip(bars2, r2_scores)):
    bar.set_height(0.7)
    # Add value labels
    ax2.text(val + 0.015, i, f'{val:.4f}', 
             va='center', ha='left', fontsize=11, fontweight='bold')

ax2.set_yticks(range(len(models)))
ax2.set_yticklabels(models, fontsize=11)
ax2.set_xlabel('R² Score', fontsize=13, fontweight='bold')
ax2.set_title('Model Performance: R² Score\n(Higher is Better)', 
              fontsize=15, fontweight='bold', pad=20)
ax2.axvline(x=0.9, color='gray', linestyle='--', linewidth=1.5, alpha=0.4, label='Excellence Threshold')
ax2.grid(axis='x', alpha=0.2, linestyle='-', linewidth=0.5)
ax2.set_xlim(0, 1.05)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Add main title
fig.suptitle('Machine Learning Model Performance Comparison\n9 Models Evaluated on NYC Taxi Fare Prediction', 
             fontsize=17, fontweight='bold', y=0.98)

plt.tight_layout()
plt.savefig('docs/images/model_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Model comparison chart saved!")

# Create stunning prediction accuracy visualization
fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')

test_cases = ['Short Trip\n(2 miles)', 'Medium Trip\n(5 miles)', 'Long Trip\n(10 miles)', 
              'Airport Run\n(15 miles)', 'Late Night\n(1.5 miles)', 'Very Long\n(25 miles)']
expected_min = [10, 18, 30, 45, 8, 70]
expected_max = [12, 22, 35, 50, 10, 80]
predicted = [10.20, 22.78, 41.48, 61.16, 8.79, 94.74]

x = np.arange(len(test_cases))
width = 0.5

# Calculate expected ranges
expected_mid = [(mi + ma)/2 for mi, ma in zip(expected_min, expected_max)]
expected_err = [(ma - mi)/2 for mi, ma in zip(expected_min, expected_max)]

# Plot expected ranges as shaded areas with gradient
for i, (mid, err) in enumerate(zip(expected_mid, expected_err)):
    ax.fill_between([i-width/2, i+width/2], 
                    [mid-err, mid-err], 
                    [mid+err, mid+err],
                    alpha=0.25, color='#3498db', label='Expected Range' if i == 0 else '')
    # Add center line
    ax.plot([i-width/2, i+width/2], [mid, mid], 
           color='#2980b9', linewidth=2, alpha=0.5)

# Plot predictions as large diamonds
scatter = ax.scatter(x, predicted, color='#e74c3c', s=300, zorder=5, 
                    marker='D', label='Model Prediction', 
                    edgecolors='white', linewidths=3)

# Add accuracy indicators
for i, (pred, e_min, e_max) in enumerate(zip(predicted, expected_min, expected_max)):
    if e_min <= pred <= e_max:
        ax.text(i, pred + 4, '✓ Within Range', fontsize=10, ha='center', 
               color='green', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
    else:
        diff = min(abs(pred - e_min), abs(pred - e_max))
        ax.text(i, pred + 4, f'±${diff:.0f}', fontsize=10, ha='center', 
               color='orange', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))
    
    # Add value labels
    ax.text(i, pred - 4, f'${pred:.2f}', fontsize=11, ha='center', 
           fontweight='bold', color='#c0392b')

ax.set_ylabel('Fare Amount ($)', fontsize=14, fontweight='bold')
ax.set_xlabel('Trip Type', fontsize=14, fontweight='bold')
ax.set_title('Fare Prediction Accuracy: Model vs Expected Ranges\nRandom Forest Model (95% Accuracy)', 
            fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(test_cases, fontsize=11, fontweight='600')
ax.legend(fontsize=12, loc='upper left', framealpha=0.9)
ax.grid(axis='y', alpha=0.2, linestyle='-', linewidth=0.5)
ax.set_ylim(0, max(predicted) + 15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('docs/images/prediction_accuracy.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Prediction accuracy chart saved!")

plt.close('all')
print("\n🎨 All visualizations created with professional styling!")
