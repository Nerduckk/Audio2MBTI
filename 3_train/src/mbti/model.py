import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold

class MBTIStackingEnsemble:
    def __init__(self, target_labels, label_names):
        self.target_labels = target_labels
        self.label_names = label_names
        self.models = {}
        self.thresholds = {}
        
    def _choose_threshold(self, y_val, val_probs):
        best_t = 0.5
        best_acc = -1.0
        for t in np.arange(0.1, 0.9, 0.0025):
            acc = accuracy_score(y_val, (val_probs >= t).astype(int))
            if acc > best_acc:
                best_acc = acc
                best_t = t
        return best_t

    def generate_oof_probs(self, X, y_all):
        """Layer 1: Generate Cross-Label Out-Of-Fold probabilities."""
        print("Layer 1: Generating OOF Cross-Label Probabilities...")
        oof_probs = np.zeros((X.shape[0], len(self.target_labels)))
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for dim_idx in range(len(self.target_labels)):
            y = y_all[:, dim_idx].astype(int)
            for train_idx, val_idx in skf.split(X, y):
                X_tr, X_va = X[train_idx], X[val_idx]
                y_tr, y_va = y[train_idx], y[val_idx]
                
                model = xgb.XGBClassifier(
                    n_estimators=800, learning_rate=0.08, max_depth=6,
                    tree_method="hist", random_state=42, n_jobs=-1, verbosity=0
                )
                model.fit(X_tr, y_tr)
                oof_probs[val_idx, dim_idx] = model.predict_proba(X_va)[:, 1]
        return oof_probs

    def train_layer2(self, X_final, y_all, dim_idx, config):
        """Layer 2: Train final meta-model for one dimension."""
        y = y_all[:, dim_idx].astype(int)
        
        # Test split
        idx_train_val, idx_test = train_test_split(np.arange(len(y)), test_size=0.15, random_state=42, stratify=y)
        # Validation split
        idx_train, idx_val = train_test_split(idx_train_val, test_size=0.15, random_state=42, stratify=y[idx_train_val])
        
        X_train, X_val, X_test = X_final[idx_train], X_final[idx_val], X_final[idx_test]
        y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

        model = xgb.XGBClassifier(
            n_estimators=config.get('n_estimators', 4000),
            learning_rate=config.get('learning_rate', 0.004),
            max_depth=config.get('max_depth', 12),
            subsample=0.85,
            colsample_bytree=0.6,
            random_state=42,
            eval_metric="logloss",
            early_stopping_rounds=180,
            tree_method="hist",
            max_bin=128, 
            n_jobs=-1
        )

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        # Calibration
        threshold = self._choose_threshold(y_val, model.predict_proba(X_val)[:, 1])
        self.thresholds[self.target_labels[dim_idx]] = threshold
        self.models[self.target_labels[dim_idx]] = model
        
        # Eval
        test_probs = model.predict_proba(X_test)[:, 1]
        preds = (test_probs >= threshold).astype(int)
        
        return {
            "accuracy": accuracy_score(y_test, preds),
            "f1_macro": f1_score(y_test, preds, average='macro'),
            "report": classification_report(y_test, preds, output_dict=True)
        }
