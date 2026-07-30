import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv("preprocessed_data.csv")

major_cols = [
    'boolean_Computer Science',
    'boolean_Information Technology',
    'boolean_Electrical Engineering',
    'boolean_Mechanical Engineering',
    'boolean_Electronics and Communication'
]
feature_cols = ['age', 'gpa'] + major_cols
X = df[feature_cols]
y = df['placement_status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

weight_ratios = np.arange(1, 15.01, 0.1)

type1_errors = []  # False Positive Rate: predicted Placed, actually Not Placed
type2_errors = []  # False Negative Rate: predicted Not Placed, actually Placed
is_degenerate = []  # True if the model only ever predicts a single class on the test set

for w in weight_ratios:
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight={0: w, 1: 1})
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

    type1 = fp / (fp + tn)
    type2 = fn / (fn + tp)

    type1_errors.append(type1)
    type2_errors.append(type2)
    is_degenerate.append(len(set(pred)) == 1)

type1_errors = np.array(type1_errors)
type2_errors = np.array(type2_errors)
is_degenerate = np.array(is_degenerate)
balanced_error_rate = (type1_errors + type2_errors) / 2

results_df = pd.DataFrame({
    'weight_ratio': weight_ratios,
    'type1_error_fpr': type1_errors,
    'type2_error_fnr': type2_errors,
    'balanced_error_rate': balanced_error_rate,
    'degenerate_single_class_model': is_degenerate
})
results_df.head(10)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(weight_ratios, type1_errors, label='Type I Error (False Positive Rate)', color='tab:red')
ax.plot(weight_ratios, type2_errors, label='Type II Error (False Negative Rate)', color='tab:blue')
ax.plot(weight_ratios, balanced_error_rate, label='Balanced Error Rate = (Type I + Type II) / 2',
        color='tab:green', linewidth=2, linestyle='--')

shaded_label_used = False
for w, deg in zip(weight_ratios, is_degenerate):
    if deg:
        ax.axvspan(w - 0.05, w + 0.05, color='gray', alpha=0.15,
                   label='Degenerate (single-class) region' if not shaded_label_used else None)
        shaded_label_used = True

ax.set_xlabel('Class Weight Ratio (Not Placed : Placed)')
ax.set_ylabel('Error Rate')
ax.set_title('Type I vs Type II Error Across Class Weight Ratios')
ax.legend()
ax.grid(True)
plt.show()

n_degenerate = is_degenerate.sum()
if n_degenerate > 0:
    deg_weights = weight_ratios[is_degenerate]
    print(f"Excluded {n_degenerate} degenerate (single-class) weight(s) from the search "
          f"(range: {deg_weights.min():.1f}-{deg_weights.max():.1f})\n")

masked_ber = np.where(is_degenerate, np.inf, balanced_error_rate)
optimal_idx = np.argmin(masked_ber)
optimal_weight = weight_ratios[optimal_idx]

print(f"Optimal weight ratio (minimum Balanced Error Rate): {optimal_weight:.2f}")
print(f"  Type I error (FPR): {type1_errors[optimal_idx]:.3f}")
print(f"  Type II error (FNR): {type2_errors[optimal_idx]:.3f}")
print(f"  Balanced Error Rate: {balanced_error_rate[optimal_idx]:.3f}")

fig, ax = plt.subplots(figsize=(10, 6))
plot_ber = np.where(is_degenerate, np.nan, balanced_error_rate)
ax.plot(weight_ratios, plot_ber, color='tab:green', label='Balanced Error Rate (non-degenerate)')
ax.axvline(optimal_weight, color='black', linestyle='--',
           label=f'Minimum BER at weight ≈ {optimal_weight:.2f}')
ax.set_xlabel('Class Weight Ratio (Not Placed : Placed)')
ax.set_ylabel('Balanced Error Rate')
ax.set_title('Balanced Error Rate vs. Weight Ratio')
ax.legend()
ax.grid(True)
plt.show()

best_model = LogisticRegression(max_iter=1000, random_state=42, class_weight={0: optimal_weight, 1: 1})
best_model.fit(X_train, y_train)
best_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, best_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Placed (0)', 'Placed (1)'])
disp.plot(cmap='Blues', values_format='d')
plt.title(f'Confusion Matrix at Optimal Weight ({optimal_weight:.2f} : 1)')
plt.grid(False)
plt.show()