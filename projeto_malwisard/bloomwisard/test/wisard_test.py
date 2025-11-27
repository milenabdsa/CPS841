import numpy as np
import sys
import pickle
from datetime import datetime
sys.path.append("../") 
import bloomwisard as wnn

# FLAG: True = teste com Malevis dataset, False = teste original
USE_MALEVIS = True

# Criar nome do arquivo de resultado com data e hora
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_filename = f"wisard_results_{timestamp}.txt"

# Redirecionar output para arquivo e console
import io
output_buffer = io.StringIO()

def log_print(*args, **kwargs):
    """Print para console e arquivo"""
    print(*args, **kwargs)
    print(*args, **kwargs, file=output_buffer)

# Compatibilidade numpy
class NumpyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)

if USE_MALEVIS:
    log_print("Loading Malevis dataset...")
    
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
    bloom_size = 1000
    
    log_print(f"Shape: {X_train.shape}, Classes: {num_classes}")
    log_print(f"Parameters: tuple_size={tuple_size}, bloom_size={bloom_size}")
    
    # Treinar BloomWisard
    log_print("\nTraining BloomWisard on Malevis...")
    start_time = datetime.now()
    bwisard = wnn.BloomWisard(num_inputs, tuple_size, num_classes, bloom_size)
    bwisard.train(X_train.tolist(), y_train.tolist())
    train_time = (datetime.now() - start_time).total_seconds()
    log_print(f"Training time: {train_time:.2f} seconds")
    
    # Avaliar
    log_print("\nEvaluating...")
    start_time = datetime.now()
    predictions = bwisard.rank(X_val.tolist())
    eval_time = (datetime.now() - start_time).total_seconds()
    log_print(f"Evaluation time: {eval_time:.2f} seconds")
    
    # O rank retorna um numpy array de inteiros (a classe predita)
    correct = sum(1 for pred, true in zip(predictions, y_val) if pred == true)
    log_print(f"\nAccuracy: {correct}/{len(y_val)} ({100*correct/len(y_val):.2f}%)")
    
    # Capturar info
    print("\nModel Info:")
    bwisard.info()

else:
    # Teste original
    log_print("Running original test...")
    a = np.array([0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1], dtype=bool)
    b = np.array([0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1], dtype=bool)

    #Discriminator
    disc = wnn.Discriminator(20, 4)
    disc.train(a)
    log_print(disc.rank(a), disc.rank(b))

    #BloomDiscriminator
    bloom_disc = wnn.BloomDiscriminator(20, 4, 1000)
    bloom_disc.train(a)
    log_print(bloom_disc.rank(a), bloom_disc.rank(b))
    bloom_disc.info()

    #Wisard
    log_print("Wisard")
    wisard2 = wnn.Wisard(20, 4, 2)
    c = [a, b]
    wisard2.train(c, [0, 1])
    log_print(wisard2.rank(c))
    wisard2.info()

    #BloomWisard
    log_print("Bloom Wisard")
    bwisard = wnn.BloomWisard(20, 4, 2, 1000)
    bwisard.train(c, [0, 1])
    log_print(bwisard.rank(c))
    bwisard.info()

    log_print("Dict Wisard")
    wisard = wnn.DictWisard(20, 4, 2)
    wisard.train(c, [0, 1])
    log_print(wisard.rank(c))
    wisard.info()

log_print("\nDone")

# Salvar resultados em arquivo
with open(result_filename, 'w', encoding='utf-8') as f:
    f.write(output_buffer.getvalue())

print(f"\n{'='*60}")
print(f"Results saved to: {result_filename}")
print(f"{'='*60}")
