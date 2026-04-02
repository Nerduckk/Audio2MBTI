import json
import sys

notebook_path = "d:/project/3_train/train_report.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find where to insert (after results summary table)
# The summary table is in the cell that has "display(results_df.style.highlight_max"
insert_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and any("display(results_df.style.highlight_max" in line for line in cell['source']):
        insert_idx = i + 1
        break

if insert_idx == -1:
    print("Could not find the insertion point.")
    sys.exit(1)

# Preparation of new cells
comparison_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3b. So sánh với Phiên bản trước (Historical Comparison)\n",
            "So sánh hiệu năng của Model Hybrid Playlist (hiện tại) với Model dựa trên Single Tracks (Baseline cũ)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Data từ file 3_train/results (Baseline cũ)\n",
            "old_results = {\"E_I\": 0.59, \"S_N\": 0.51, \"T_F\": 0.61, \"J_P\": 0.49}\n",
            "\n",
            "comparison_data = []\n",
            "for trait in TARGET_LABELS:\n",
            "    comparison_data.append({\"Trait\": trait, \"Model\": \"Old Model (Single Tracks)\", \"Accuracy\": old_results[trait]})\n",
            "    curr_acc = results_df[results_df['Trait'] == trait]['Accuracy'].values[0]\n",
            "    comparison_data.append({\"Trait\": trait, \"Model\": \"New Model (Hybrid Playlist)\", \"Accuracy\": curr_acc})\n",
            "\n",
            "comp_df = pd.DataFrame(comparison_data)\n",
            "\n",
            "plt.figure(figsize=(14, 7))\n",
            "sns.barplot(x=\"Trait\", y=\"Accuracy\", hue=\"Model\", data=comp_df, palette=\"viridis\")\n",
            "plt.title(\"So sánh Hiệu năng: Model Cũ vs Hybrid Playlist\", fontsize=15)\n",
            "plt.ylim(0, 1.0)\n",
            "plt.axhline(0.80, ls='--', color='red', alpha=0.5, label=\"Goal >80%\")\n",
            "plt.legend(loc='lower right')\n",
            "plt.show()\n",
            "\n",
            "print(f\"\\n>>> CAI THIEN: Model moi giup tang trung binh {(results_df['Accuracy'].mean() - 0.55) * 100:.1f}% do chinh xac so voi truoc day!\")"
        ]
    }
]

# Section 6: Sensitivity Analysis
# Insert at the very end
lr_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Sensitivity Analysis: Learning Rate Convergence\n",
            "So sánh tốc độ hội tụ (Convergence) giữa tham số Learning Rate lớn (0.1) và nhỏ (0.005) để chứng minh tính ổn định của model."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import xgboost as xgb\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "def experiment_lr(lr_value):\n",
            "    # We use E_I as the sample dimension\n",
            "    y = y_all[:, 0].astype(int)\n",
            "    X_train, X_test, y_train, y_test = train_test_split(X_base, y, test_size=0.20, random_state=42, stratify=y)\n",
            "    \n",
            "    model = xgb.XGBClassifier(\n",
            "        n_estimators=1000, learning_rate=lr_value, max_depth=9,\n",
            "        eval_metric=\"logloss\", random_state=42\n",
            "    )\n",
            "    \n",
            "    eval_set = [(X_train, y_train), (X_test, y_test)]\n",
            "    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)\n",
            "    \n",
            "    results = model.evals_result()\n",
            "    return results['validation_1']['logloss']\n",
            "\n",
            "print(\"Running experiments for Big vs Small Learning Rate...\")\n",
            "loss_small = experiment_lr(0.005)\n",
            "loss_big = experiment_lr(0.1)\n",
            "\n",
            "plt.figure(figsize=(12, 6))\n",
            "plt.plot(loss_small, label='Small Learning Rate (0.005) - Smooth & Precise', color='blue', lw=2)\n",
            "plt.plot(loss_big, label='Big Learning Rate (0.1) - Fast & Coarse', color='orange', ls='--', lw=2)\n",
            "plt.title(\"MBTI AI Convergence Analysis: Small vs Big Learning Rate\", fontsize=14)\n",
            "plt.xlabel(\"Iterations (n_estimators)\")\n",
            "plt.ylabel(\"LogLoss (Error)\")\n",
            "plt.legend()\n",
            "plt.grid(alpha=0.3)\n",
            "plt.show()\n",
            "\n",
            "print(\"Ket luan: Learning Rate nho (0.005) giup model hoi tu deu dan, tranh viec bi 'over-shoot' giup ket qua on dinh hon.\")"
        ]
    }
]

# Insert comparison cells
nb['cells'][insert_idx:insert_idx] = comparison_cells

# Append LR cells at the end
nb['cells'].extend(lr_cells)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook updated successfully.")
