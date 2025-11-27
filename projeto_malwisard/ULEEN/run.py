import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import pickle
import numpy as np
from time import perf_counter
import signal
import sys

from software_model.model import BackpropMultiWiSARD

# Variável global para controlar interrupção do treinamento
Abort_Training = False

def sigint_handler(signum, frame):
    global Abort_Training
    if not Abort_Training:
        print("Will abort training at end of epoch")
        Abort_Training = True
    else:
        sys.exit("Quitting immediately on second SIGINT")

# Run inference usando dataset (validação ou teste)
def run_inference(model, dset_loader, collect_submodel_accuracies=False):
    total = 0
    correct = 0
    device = next(model.parameters()).device
    if collect_submodel_accuracies:
        submodel_correct = torch.zeros(len(model.models), device=device)
    for features, labels in dset_loader:
        features, labels = features.to(device), labels.to(device)
        model_results = model(features)
        if model_results.ndim == 3:
            outputs = model_results.sum(axis=0)
        else:
            outputs = model_results

        _, predicted = torch.max(outputs.data, axis=1)
        if collect_submodel_accuracies:
            submodel_predicted = torch.argmax(model_results.data,
                                              axis=2).detach()
        total += labels.size(0)
        correct += (predicted == labels).sum()
        if collect_submodel_accuracies:
            submodel_correct += (submodel_predicted == labels).sum(axis=1)

    if collect_submodel_accuracies:
        return total, correct, submodel_correct
    else:
        return total, correct

# Treinar modelo ULEEN (Etapa 1)
def train_model(
    model, train_loader, val_loader, num_epochs=100,
    learning_rate=1e-3, decay_lr=False, device=None
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    global Abort_Training
    Abort_Training = False
    old_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, sigint_handler)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    if decay_lr:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=(num_epochs*3)//10)

    for epoch in range(num_epochs):
        start_time = perf_counter()

        train_total = 0
        train_correct = 0
        model.train()
        
        batch_count = 0
        for features, labels in train_loader:
            batch_count += 1
            if batch_count == 1:
                print(f"Processing epoch {epoch}, first batch...")
            if batch_count % 10 == 0:
                print(f"  Batch {batch_count}/{len(train_loader)}...")
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            model_results = model(features)
            outputs = model_results.sum(axis=0)
          
            _, predicted = torch.max(outputs.data, 1)

            train_total += labels.size(0)
            train_correct += (predicted == labels).sum()

            loss = sum(criterion(o, labels) for o in model_results)
            loss.backward()
            optimizer.step()
            model.clamp()

        model.eval()
        val_total, val_correct, val_submodel_correct = \
            run_inference(model, val_loader, True)
        end_time = perf_counter()

        print(f"At end of epoch {epoch}: "
              f"Train set: Correct: {train_correct}/{train_total} "
              f"({round(((100*train_correct)/train_total).item(), 3)}%); "
              f"Validation set: Correct: {val_correct}/{val_total} "
              f"({round(((100*val_correct)/val_total).item(), 3)}%)")
        submodel_accuracies = [f"{round(100*(c/val_total).item(), 3)}%"
                               for c in val_submodel_correct]
        print(f"  Submodel accuracies: {' '.join(submodel_accuracies)}; Time elapsed: {round(end_time-start_time, 2)}")
        
        if decay_lr:
            scheduler.step()
        
        if Abort_Training:
            break
    
    model.eval()
    signal.signal(signal.SIGINT, old_handler)
    
    return model

def load_malevis_data(data_path='../'):
    """
    Carrega dados do Malevis em formato pickle
    """
    # Compatibilidade com diferentes versões do numpy
    # numpy 2.x usa numpy._core, numpy 1.x usa numpy.core
    import sys
    
    class NumpyUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # Redirecionar numpy._core para numpy.core para compatibilidade
            if module.startswith('numpy._core'):
                module = module.replace('numpy._core', 'numpy.core')
            return super().find_class(module, name)
    
    # Carregar dados de treino
    with open(f'{data_path}thermometer12_train_X.p', 'rb') as f:
        X_train_raw = NumpyUnpickler(f).load()
    with open(f'{data_path}thermometer12_train_y.p', 'rb') as f:
        y_train_raw = NumpyUnpickler(f).load()
    
    # Carregar dados de validação
    with open(f'{data_path}thermometer12_val_X.p', 'rb') as f:
        X_val_raw = NumpyUnpickler(f).load()
    with open(f'{data_path}thermometer12_val_y.p', 'rb') as f:
        y_val_raw = NumpyUnpickler(f).load()
    
    # Converter para numpy arrays se necessário
    if isinstance(X_train_raw, list):
        X_train = np.array(X_train_raw, dtype=np.float32)
    else:
        X_train = X_train_raw.astype(np.float32)
    
    if isinstance(X_val_raw, list):
        X_val = np.array(X_val_raw, dtype=np.float32)
    else:
        X_val = X_val_raw.astype(np.float32)
    
    # Processar labels - podem ser strings que precisam ser convertidas para inteiros
    def process_labels(labels_raw):
        if isinstance(labels_raw, list):
            labels = np.array(labels_raw)
        else:
            labels = labels_raw
        
        # Garantir que seja 1D
        if labels.ndim > 1:
            labels = labels.flatten()
        
        # Se os labels são strings, mapear para inteiros
        if labels.dtype.kind in ['U', 'S', 'O']:  # Unicode, bytes, ou object
            unique_labels = np.unique(labels)
            label_to_int = {label: idx for idx, label in enumerate(sorted(unique_labels))}
            print(f"Label mapping: {label_to_int}")
            labels = np.array([label_to_int[label] for label in labels], dtype=np.int64)
        else:
            labels = labels.astype(np.int64)
        
        return labels
    
    y_train = process_labels(y_train_raw)
    y_val = process_labels(y_val_raw)
    
    print(f"Dataset shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Dataset shapes - X_val: {X_val.shape}, y_val: {y_val.shape}")
    
    # Converter para tensors
    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val)
    
    return X_train_t, y_train_t, X_val_t, y_val_t

if __name__ == "__main__":
    # Configuração
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Carregar dados Malevis
    print("Loading Malevis dataset...")
    X_train, y_train, X_val, y_val = load_malevis_data()
    
    # Obter informações do dataset
    inputs = X_train.shape[1]
    classes = int(y_train.max() + 1)
    print(f"Num inputs/classes: {inputs}/{classes}")
    
    # Criar TensorDatasets
    train_set = TensorDataset(X_train, y_train)
    val_set = TensorDataset(X_val, y_val)
    
    # Configurações do modelo ULEEN - Etapa 1
    configs = [(8, 256, 4), (16, 512, 4), (32, 1024, 4)]
    encoding_bits = 12  # Thermometer encoding de 12 bits
    dropout_p = 0.1
    batch_size = 32
    num_epochs = 50
    learning_rate = 1e-3
    decay_lr = False
    
    print("Model configs:", configs)
    print(f"Encoding bits: {encoding_bits}")
    print(f"Dropout: {dropout_p}")
    print(f"Batch size: {batch_size}")
    
    # Criar data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True)
    
    # Criar modelo ULEEN
    print("\nInitializing ULEEN model...")
    print(f"Model will have {len(configs)} submodels")
    model = BackpropMultiWiSARD(
        inputs=inputs,
        classes=classes,
        configs=configs,
        encoding_bits=encoding_bits,
        dropout_p=dropout_p
    )
    print("Model initialized successfully!")
    
    # Etapa 1: Treinar modelo ULEEN
    print(f"\n{'='*60}")
    print("Step 1/3: Train ULEEN model")
    print(f"{'='*60}\n")
    print("Starting training...")
    
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        decay_lr=decay_lr,
        device=device
    )
    
    print("Training completed!")
    
    # Avaliar no conjunto de validação final
    print("\nFinal validation accuracy:")
    val_total, val_correct = run_inference(model, val_loader)
    print(f"Validation set: Correct: {val_correct}/{val_total} "
          f"({round(((100*val_correct)/val_total).item(), 3)}%)")
    
    # Salvar modelo
    model = model.to("cpu")
    model_fname = "malevis_uleen_model.pt"
    torch.save(model, model_fname)
    print(f"\nModel saved to {model_fname}")
    print("\nStep 1 complete! Model is ready for pruning (Step 2) and finalization (Step 3).")