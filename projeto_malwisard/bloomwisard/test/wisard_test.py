import numpy as np
import sys
import pickle
sys.path.append("../") 
from core import wnn

# FLAG: True = teste com Malevis dataset, False = teste original
USE_MALEVIS = True

# Compatibilidade numpy
class NumpyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)

if USE_MALEVIS:
    print("Loading Malevis dataset...")
    
    # Carregar dados
    with open('../../thermometer12_train_X.p', 'rb') as f:
        X_train = np.array(NumpyUnpickler(f).load(), dtype=bool)
    with open('../../thermometer12_train_y.p', 'rb') as f:
        y_train_raw = NumpyUnpickler(f).load()
    with open('../../thermometer12_val_X.p', 'rb') as f:
        X_val = np.array(NumpyUnpickler(f).load(), dtype=bool)
    with open('../../thermometer12_val_y.p', 'rb') as f:
        y_val_raw = NumpyUnpickler(f).load()
    
    # Mapear labels
    labels_unique = sorted(set(y_train_raw))
    label_map = {label: i for i, label in enumerate(labels_unique)}
    y_train = np.array([label_map[l] for l in y_train_raw])
    y_val = np.array([label_map[l] for l in y_val_raw])
    
    num_inputs = X_train.shape[1]
    num_classes = len(label_map)
    tuple_size = 8
    
    print(f"Shape: {X_train.shape}, Classes: {num_classes}")
    
    # Treinar BloomWisard
    print("\nTraining BloomWisard on Malevis...")
    bwisard = wnn.BloomWisard(num_inputs, tuple_size, num_classes, 1000)
    bwisard.train(X_train.tolist(), y_train.tolist())
    
    # Avaliar
    print("Evaluating...")
    predictions = bwisard.rank(X_val.tolist())
    correct = sum(1 for pred, true in zip(predictions, y_val) if pred[0] == true)
    print(f"Accuracy: {correct}/{len(y_val)} ({100*correct/len(y_val):.2f}%)")
    
    bwisard.info()

else:
    # Teste original
    print("Running original test...")
    a = np.array([0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1], dtype=bool)
    b = np.array([0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1], dtype=bool)

    #Discriminator
    disc = wnn.Discriminator(20, 4)
    disc.train(a)
    print(disc.rank(a), disc.rank(b))

    #BloomDiscriminator
    bloom_disc = wnn.BloomDiscriminator(20, 4, 1000)
    bloom_disc.train(a)
    print(bloom_disc.rank(a), bloom_disc.rank(b))
    bloom_disc.info()

    #Wisard
    print("Wisard")
    wisard2 = wnn.Wisard(20, 4, 2)
    c = [a, b]
    wisard2.train(c, [0, 1])
    print(wisard2.rank(c))
    wisard2.info()

    #BloomWisard
    print("Bloom Wisard")
    bwisard = wnn.BloomWisard(20, 4, 2, 1000)
    bwisard.train(c, [0, 1])
    print(bwisard.rank(c))
    bwisard.info()

    print("Dict Wisard")
    wisard = wnn.DictWisard(20, 4, 2)
    wisard.train(c, [0, 1])
    print(wisard.rank(c))
    wisard.info()

print("Done")
