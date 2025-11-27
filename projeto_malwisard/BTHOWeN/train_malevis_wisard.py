import pickle
import numpy as np
import sys
from scipy.stats import norm
sys.path.insert(0, './software_model')
from wisard import WiSARD

# FLAG: True = usar dados já binarizados (thermometer12_*.p)
#       False = usar dados pre-binarização (y26_*.p) e aplicar thermometer encoding
USE_BINARIZED_DATA = True

# Compatibilidade numpy
class NumpyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)

def apply_thermometer_encoding(data, bits_per_input=12):
    """Aplica thermometer encoding nos dados (igual train_swept_models.py)"""
    print(f"Applying thermometer encoding ({bits_per_input} bits)...")
    std_skews = [norm.ppf((i+1)/(bits_per_input+1)) for i in range(bits_per_input)]
    mean_data = data.mean(axis=0)
    std_data = data.std(axis=0)
    
    binarizations = []
    for skew in std_skews:
        binarizations.append((data >= mean_data + (skew * std_data)).astype(np.uint8))
    
    return np.concatenate(binarizations, axis=1)

# Carregar dados
print("Loading data...")
if USE_BINARIZED_DATA:
    print("Using pre-binarized data (thermometer12_*.p)")
    with open('../thermometer12_train_X.p', 'rb') as f:
        X_train = np.array(NumpyUnpickler(f).load(), dtype=np.uint8)
    with open('../thermometer12_train_y.p', 'rb') as f:
        y_train_raw = NumpyUnpickler(f).load()
    with open('../thermometer12_val_X.p', 'rb') as f:
        X_val = np.array(NumpyUnpickler(f).load(), dtype=np.uint8)
    with open('../thermometer12_val_y.p', 'rb') as f:
        y_val_raw = NumpyUnpickler(f).load()
else:
    print("Using raw data (y26_*.p) and applying thermometer encoding")
    with open('../y26.p', 'rb') as f:
        X_train_raw = np.array(NumpyUnpickler(f).load(), dtype=np.float32)
    with open('../y26_val.p', 'rb') as f:
        X_val_raw = np.array(NumpyUnpickler(f).load(), dtype=np.float32)
    
    # Aplicar thermometer encoding
    X_train = apply_thermometer_encoding(X_train_raw, bits_per_input=12)
    X_val = apply_thermometer_encoding(X_val_raw, bits_per_input=12)
    
    # Labels estão nos arquivos separados
    with open('../thermometer12_train_y.p', 'rb') as f:
        y_train_raw = NumpyUnpickler(f).load()
    with open('../thermometer12_val_y.p', 'rb') as f:
        y_val_raw = NumpyUnpickler(f).load()

# Mapear labels string para int
labels_unique = sorted(set(y_train_raw))
label_map = {label: i for i, label in enumerate(labels_unique)}
y_train = np.array([label_map[l] for l in y_train_raw], dtype=np.int64)
y_val = np.array([label_map[l] for l in y_val_raw], dtype=np.int64)

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Classes: {len(label_map)}")

# Criar e treinar modelo
print("\nTraining BTHOWeN/WiSARD...")
model = WiSARD(X_train.shape[1], len(label_map), unit_inputs=8, unit_entries=256, unit_hashes=4)

for i in range(len(X_train)):
    model.train(X_train[i], y_train[i])
    if (i+1) % 1000 == 0:
        print(f"  {i+1}/{len(X_train)}")

# Encontrar melhor bleaching (igual train_swept_models.py)
print("\nFinding best bleaching...")
max_bleach = max(f.data.max() for d in model.discriminators for f in d.filters)
best_bleach, best_acc = 1, 0

for bleach in range(1, max_bleach+1):
    model.set_bleaching(bleach)
    correct = sum(y_val[i] in model.predict(X_val[i]) for i in range(len(X_val)))
    acc = 100*correct/len(X_val)
    if acc > best_acc:
        best_bleach, best_acc = bleach, acc

print(f"Best bleach: {best_bleach}, Accuracy: {best_acc:.2f}%")
