# ultimate_aml_stacking_pipeline_fixed_v2.py
import os
import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

data_path = r"C:\Alex The Great\Project\medai-env\datasets\Microarray Innovations in LEukemia\GSE13159_RAW\FINAL_balanced_aml_expression.csv"
labels_path = r"C:\Alex The Great\Project\medai-env\datasets\Microarray Innovations in LEukemia\GSE13159_RAW\FINAL_balanced_aml_metadata.csv"

print("Loading data...")
X_df = pd.read_csv(data_path, index_col=0).T
X = X_df.values.astype('float32')
meta = pd.read_csv(labels_path)
y_text = meta['subtype'].values

subtype_mapping = {
    'AML_CK': 0, 'AML_FLT3_ITD': 1, 'AML_inv_16': 2,
    'AML_M3_APL': 3, 'AML_NK': 4, 'AML_other': 5, 'AML_t_8_21': 6
}
y = np.array([subtype_mapping[s] for s in y_text])
num_classes = len(subtype_mapping)

print(f"Original data shape: {X.shape}")
print("Class distribution:", {k: np.sum(y == v) for k, v in subtype_mapping.items()})

X_train_full, X_test_full, y_train_full, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=SEED
)
print("Split sizes - train:", X_train_full.shape, "test:", X_test_full.shape)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test_full)

K = 3000
selector = SelectKBest(f_classif, k=min(K, X_train_scaled.shape[1]))
X_train_sel = selector.fit_transform(X_train_scaled, y_train_full)
X_test_sel = selector.transform(X_test_scaled)
selected_indices = selector.get_support(indices=True)
print(f"Selected {X_train_sel.shape[1]} features out of {X.shape[1]}")

joblib.dump(scaler, "scaler.pkl")
joblib.dump(selected_indices, "selected_indices.pkl")

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_full),
    y=y_train_full
)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)

def augment_batch(X_batch, noise_std=0.005, feature_keep_prob=0.98):
    noise = np.random.normal(0, noise_std, X_batch.shape).astype(np.float32)
    X_batch = X_batch + noise
    mask = np.random.binomial(1, feature_keep_prob, X_batch.shape).astype(np.float32)
    return X_batch * mask

def weighted_data_generator(X, y, class_weights, batch_size=32, augment=False, shuffle=True):
    n = len(X)
    indices = np.arange(n)
    
    sample_weights = np.array([class_weights[label] for label in y])
    
    while True:
        if shuffle:
            shuffled_indices = np.random.permutation(n)
            X_shuffled = X[shuffled_indices]
            y_shuffled = y[shuffled_indices]
            weights_shuffled = sample_weights[shuffled_indices]
        else:
            X_shuffled = X
            y_shuffled = y
            weights_shuffled = sample_weights
        
        for start_idx in range(0, n, batch_size):
            end_idx = min(start_idx + batch_size, n)
            batch_indices = slice(start_idx, end_idx)
            
            X_batch = X_shuffled[batch_indices]
            y_batch = y_shuffled[batch_indices]
            weight_batch = weights_shuffled[batch_indices]
            
            if augment:
                X_batch = augment_batch(X_batch)
            
            yield (X_batch.reshape(-1, X_batch.shape[1], 1), 
                   y_batch, 
                   weight_batch)

def create_cnn(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv1D(128, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    
    x = layers.Conv1D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs, outputs, name='cnn')

def create_dense(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Flatten()(inputs)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs, outputs, name='dense')

def create_transformer(input_shape, num_classes, head_size=256, num_heads=4, ff_dim=4):
    inputs = layers.Input(shape=input_shape)
    
    x = inputs
    
    for _ in range(2):
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        attn_output = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=0.2
        )(x, x)
        x = layers.Add()([x, attn_output])
        
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        ffn = layers.Dense(ff_dim * head_size, activation="relu")(x)
        ffn = layers.Dropout(0.2)(ffn)
        ffn = layers.Dense(head_size)(ffn)
        x = layers.Add()([x, ffn])
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    return models.Model(inputs, outputs, name='transformer')

def compile_model(model, lr=1e-3):
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy'],
        weighted_metrics=['sparse_categorical_accuracy']
    )
    return model

class CosineAnnealingScheduler(callbacks.Callback):
    def __init__(self, T_max, eta_max=1e-3, eta_min=1e-6, verbose=0):
        super().__init__()
        self.T_max = T_max
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.verbose = verbose
        self.epoch = 0
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        lr = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (
            1 + np.cos(np.pi * self.epoch / self.T_max)
        )
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
        if self.verbose > 0:
            print(f'\nEpoch {self.epoch}: Learning rate is {lr:.6f}')

def train_model_with_weights(model, X_train, y_train, X_val, y_val, class_weights, 
                            batch_size=32, epochs=100, callbacks_list=None):
    steps_per_epoch = math.ceil(len(X_train) / batch_size)
    
    train_gen = weighted_data_generator(X_train, y_train, class_weights, 
                                      batch_size=batch_size, augment=True)
    
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=(X_val.reshape(-1, X_val.shape[1], 1), y_val),
        epochs=epochs,
        callbacks=callbacks_list,
        verbose=1
    )
    
    return history

def train_model_simple(model, X_train, y_train, X_val, y_val, class_weights, 
                      batch_size=32, epochs=100):
    steps_per_epoch = math.ceil(len(X_train) / batch_size)
    
    train_gen = weighted_data_generator(X_train, y_train, class_weights, 
                                      batch_size=batch_size, augment=True)
    
    callbacks_list = [
        callbacks.EarlyStopping(
            patience=15,
            restore_best_weights=True,
            monitor='val_sparse_categorical_accuracy',
            mode='max',
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            f'{model.name}_fold.keras',
            save_best_only=True,
            monitor='val_sparse_categorical_accuracy',
            mode='max',
            verbose=0
        )
    ]
    
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=(X_val.reshape(-1, X_val.shape[1], 1), y_val),
        epochs=epochs,
        callbacks=callbacks_list,
        verbose=1
    )
    
    return history

NFOLDS = 5
BATCH_SIZE = 16
EPOCHS = 120
skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)

X_train = X_train_sel
n_train = X_train.shape[0]

oof_preds_cnn = np.zeros((n_train, num_classes), dtype=np.float32)
oof_preds_dense = np.zeros((n_train, num_classes), dtype=np.float32)
oof_preds_transformer = np.zeros((n_train, num_classes), dtype=np.float32)

test_preds_cnn_folds = []
test_preds_dense_folds = []
test_preds_transformer_folds = []

print("\nStarting K-Fold Training...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train_full)):
    print(f"\n{'='*50}")
    print(f"FOLD {fold+1}/{NFOLDS}")
    print(f"{'='*50}")
    
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train_full[train_idx], y_train_full[val_idx]
    
    models_dict = {
        'cnn': create_cnn((X_tr.shape[1], 1), num_classes),
        'dense': create_dense((X_tr.shape[1], 1), num_classes),
        'transformer': create_transformer((X_tr.shape[1], 1), num_classes)
    }
    
    for model_name, model in models_dict.items():
        print(f"\nTraining {model_name}...")
        
        compile_model(model, lr=1e-3)
        
        history = train_model_simple(
            model, X_tr, y_tr, X_val, y_val, class_weight_dict,
            batch_size=BATCH_SIZE, epochs=EPOCHS
        )
        
        val_preds = model.predict(X_val.reshape(-1, X_val.shape[1], 1), verbose=0)
        
        if model_name == 'cnn':
            oof_preds_cnn[val_idx] = val_preds
        elif model_name == 'dense':
            oof_preds_dense[val_idx] = val_preds
        else:
            oof_preds_transformer[val_idx] = val_preds
        
        test_preds = model.predict(X_test_sel.reshape(-1, X_test_sel.shape[1], 1), verbose=0)
        
        if model_name == 'cnn':
            test_preds_cnn_folds.append(test_preds)
        elif model_name == 'dense':
            test_preds_dense_folds.append(test_preds)
        else:
            test_preds_transformer_folds.append(test_preds)

print("\nPreparing meta features...")

test_preds_cnn_avg = np.mean(test_preds_cnn_folds, axis=0)
test_preds_dense_avg = np.mean(test_preds_dense_folds, axis=0)
test_preds_transformer_avg = np.mean(test_preds_transformer_folds, axis=0)

X_meta_train = np.concatenate([
    oof_preds_cnn,
    oof_preds_dense,
    oof_preds_transformer
], axis=1)

X_meta_test = np.concatenate([
    test_preds_cnn_avg,
    test_preds_dense_avg,
    test_preds_transformer_avg
], axis=1)

print("Training meta-learner...")

try:
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    
    meta_learners = {
        'xgb': XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            eval_metric='mlogloss'
        ),
        'lgbm': LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            verbose=-1
        )
    }
    
    best_meta_learner = None
    best_score = 0
    
    for name, learner in meta_learners.items():
        print(f"Testing {name} as meta-learner...")
        learner.fit(X_meta_train, y_train_full)
        score = learner.score(X_meta_train, y_train_full)
        print(f"{name} train accuracy: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_meta_learner = learner
    
    print(f"Selected {type(best_meta_learner).__name__} as meta-learner")

except ImportError:
    print("XGBoost or LightGBM not available, using GradientBoosting...")
    best_meta_learner = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        random_state=SEED
    )
    best_meta_learner.fit(X_meta_train, y_train_full)

print("\n" + "="*60)
print("FINAL EVALUATION")
print("="*60)

y_pred_meta = best_meta_learner.predict(X_meta_test)
y_pred_meta_prob = best_meta_learner.predict_proba(X_meta_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_meta, target_names=list(subtype_mapping.keys())))

cm = confusion_matrix(y_test, y_pred_meta)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(subtype_mapping.keys()),
            yticklabels=list(subtype_mapping.keys()))
plt.title('Confusion Matrix - AML Subtypes (Stacked Ensemble)', fontsize=14)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix_stacked_improved.png', dpi=300, bbox_inches='tight')
plt.show()

y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)
try:
    roc_auc = roc_auc_score(y_test_onehot, y_pred_meta_prob, multi_class='ovr')
    print(f"ROC-AUC Score: {roc_auc:.4f}")
except Exception as e:
    print(f"ROC-AUC calculation failed: {e}")

print("\nSaving artifacts...")
joblib.dump(best_meta_learner, "best_meta_learner.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(selected_indices, "selected_indices.pkl")

with open('subtype_mapping.json', 'w') as f:
    json.dump(subtype_mapping, f)

print("All artifacts saved successfully!")