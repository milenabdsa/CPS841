import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pickle
import numpy as np
from time import perf_counter
from datetime import datetime
import sys
import os

# Adicionar o diretório ao path para importar o modelo
sys.path.append(os.path.dirname(__file__))

from software_model.model import BackpropMultiWiSARD

# Compatibilidade com diferentes versões do numpy
class NumpyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)

def load_malevis_data(data_path='../'):
    """
    Carrega dados do Malevis em formato pickle
    """
    print("Loading Malevis dataset...")
    
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
    
    # Converter para numpy arrays
    if isinstance(X_train_raw, list):
        X_train = np.array(X_train_raw, dtype=np.float32)
    else:
        X_train = X_train_raw.astype(np.float32)
    
    if isinstance(X_val_raw, list):
        X_val = np.array(X_val_raw, dtype=np.float32)
    else:
        X_val = X_val_raw.astype(np.float32)
    
    # Processar labels
    def process_labels(labels_raw):
        if isinstance(labels_raw, list):
            labels = np.array(labels_raw)
        else:
            labels = labels_raw
        
        if labels.ndim > 1:
            labels = labels.flatten()
        
        if labels.dtype.kind in ['U', 'S', 'O']:
            unique_labels = np.unique(labels)
            label_to_int = {label: idx for idx, label in enumerate(sorted(unique_labels))}
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

def run_inference(model, dset_loader, collect_submodel_accuracies=False, verbose=False):
    """
    Run inference usando dataset (validação ou teste)
    """
    total = 0
    correct = 0
    device = next(model.parameters()).device
    
    if collect_submodel_accuracies:
        submodel_correct = torch.zeros(len(model.models), device=device)
    
    all_predictions = []
    all_labels = []
    
    batch_idx = 0
    for features, labels in dset_loader:
        batch_idx += 1
        if verbose and batch_idx % 10 == 0:
            print(f"  Processing batch {batch_idx}/{len(dset_loader)}...")
        
        features, labels = features.to(device), labels.to(device)
        model_results = model(features)
        
        if model_results.ndim == 3:
            outputs = model_results.sum(axis=0)
        else:
            outputs = model_results

        _, predicted = torch.max(outputs.data, axis=1)
        
        if collect_submodel_accuracies:
            submodel_predicted = torch.argmax(model_results.data, axis=2).detach()
        
        total += labels.size(0)
        correct += (predicted == labels).sum()
        
        if collect_submodel_accuracies:
            submodel_correct += (submodel_predicted == labels).sum(axis=1)
        
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    if collect_submodel_accuracies:
        return total, correct, submodel_correct, all_predictions, all_labels
    else:
        return total, correct, all_predictions, all_labels

def compute_confusion_matrix(predictions, labels, num_classes):
    """
    Computa matriz de confusão
    """
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pred, true in zip(predictions, labels):
        confusion_matrix[true, pred] += 1
    return confusion_matrix

def compute_per_class_metrics(confusion_matrix):
    """
    Computa precisão, recall e F1-score por classe
    """
    num_classes = confusion_matrix.shape[0]
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1_score = np.zeros(num_classes)
    
    for i in range(num_classes):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp
        fn = confusion_matrix[i, :].sum() - tp
        
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) \
                      if (precision[i] + recall[i]) > 0 else 0
    
    return precision, recall, f1_score

def count_model_parameters(model):
    """
    Conta o número de parâmetros treináveis do modelo
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def get_model_info(model):
    """
    Extrai informações sobre a arquitetura do modelo
    """
    # Pegar informações do primeiro submodelo
    first_submodel = model.models[0]
    
    info = {
        'num_submodels': len(model.models),
        'input_size': first_submodel.inputs,
        'num_classes': first_submodel.classes,
        'encoding_bits': model.encoding_bits,
        'configs': []
    }
    
    for i, submodel in enumerate(model.models):
        config = {
            'submodel_id': i,
            'filter_inputs': submodel.filter_inputs,
            'filter_entries': submodel.filter_entries,
            'filter_hash_functions': submodel.filter_hash_functions,
            'filters_per_discriminator': submodel.filters_per_discriminator,
            'num_classes': submodel.classes
        }
        info['configs'].append(config)
    
    return info

def test_uleen_model(model_path, output_file=None):
    """
    Testa o modelo ULEEN salvo e gera relatório completo
    """
    # Criar nome do arquivo de saída se não fornecido
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"uleen_test_results_{timestamp}.txt"
    
    # Abrir arquivo de saída
    with open(output_file, 'w', encoding='utf-8') as f:
        def log_print(msg):
            """Print para console e arquivo"""
            print(msg)
            f.write(msg + '\n')
        
        log_print("="*80)
        log_print("ULEEN MODEL TEST REPORT")
        log_print("="*80)
        log_print(f"Test date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_print(f"Model path: {model_path}")
        log_print("")
        
        # Verificar dispositivo
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log_print(f"Using device: {device}")
        log_print("")
        
        # Carregar modelo
        log_print("Loading model...")
        try:
            model = torch.load(model_path, map_location=device)
            model.eval()
            log_print("Model loaded successfully!")
        except Exception as e:
            log_print(f"ERROR loading model: {e}")
            return
        
        log_print("")
        
        # Informações do modelo
        log_print("-"*80)
        log_print("MODEL ARCHITECTURE")
        log_print("-"*80)
        
        model_info = get_model_info(model)
        log_print(f"Number of submodels: {model_info['num_submodels']}")
        log_print(f"Input size: {model_info['input_size']}")
        log_print(f"Number of classes: {model_info['num_classes']}")
        log_print(f"Encoding bits: {model_info['encoding_bits']}")
        log_print("")
        
        log_print("Submodel configurations:")
        for config in model_info['configs']:
            log_print(f"  Submodel {config['submodel_id']}: "
                     f"filter_inputs={config['filter_inputs']}, "
                     f"filter_entries={config['filter_entries']}, "
                     f"filter_hash_functions={config['filter_hash_functions']}, "
                     f"filters_per_discriminator={config['filters_per_discriminator']}, "
                     f"num_classes={config['num_classes']}")
        log_print("")
        
        # Contar parâmetros
        total_params, trainable_params = count_model_parameters(model)
        log_print(f"Total parameters: {total_params:,}")
        log_print(f"Trainable parameters: {trainable_params:,}")
        log_print("")
        
        # Carregar dados
        log_print("-"*80)
        log_print("LOADING DATA")
        log_print("-"*80)
        X_train, y_train, X_val, y_val = load_malevis_data()
        log_print("")
        
        # Criar dataloaders
        batch_size = 32
        train_set = TensorDataset(X_train, y_train)
        val_set = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # Testar no conjunto de treino
        log_print("-"*80)
        log_print("TRAINING SET EVALUATION")
        log_print("-"*80)
        
        start_time = perf_counter()
        train_total, train_correct, train_submodel_correct, train_preds, train_labels = \
            run_inference(model, train_loader, collect_submodel_accuracies=True, verbose=True)
        train_time = perf_counter() - start_time
        
        train_accuracy = (100 * train_correct / train_total).item()
        log_print(f"\nTraining set accuracy: {train_correct}/{train_total} ({train_accuracy:.2f}%)")
        log_print(f"Inference time: {train_time:.2f} seconds")
        log_print(f"Throughput: {train_total/train_time:.2f} samples/second")
        log_print("")
        
        log_print("Submodel accuracies on training set:")
        for i, correct in enumerate(train_submodel_correct):
            acc = (100 * correct / train_total).item()
            log_print(f"  Submodel {i}: {correct}/{train_total} ({acc:.2f}%)")
        log_print("")
        
        # Testar no conjunto de validação
        log_print("-"*80)
        log_print("VALIDATION SET EVALUATION")
        log_print("-"*80)
        
        start_time = perf_counter()
        val_total, val_correct, val_submodel_correct, val_preds, val_labels = \
            run_inference(model, val_loader, collect_submodel_accuracies=True, verbose=True)
        val_time = perf_counter() - start_time
        
        val_accuracy = (100 * val_correct / val_total).item()
        log_print(f"\nValidation set accuracy: {val_correct}/{val_total} ({val_accuracy:.2f}%)")
        log_print(f"Inference time: {val_time:.2f} seconds")
        log_print(f"Throughput: {val_total/val_time:.2f} samples/second")
        log_print("")
        
        log_print("Submodel accuracies on validation set:")
        for i, correct in enumerate(val_submodel_correct):
            acc = (100 * correct / val_total).item()
            log_print(f"  Submodel {i}: {correct}/{val_total} ({acc:.2f}%)")
        log_print("")
        
        # Matriz de confusão e métricas por classe
        log_print("-"*80)
        log_print("PER-CLASS METRICS (Validation Set)")
        log_print("-"*80)
        
        num_classes = model_info['num_classes']
        confusion_matrix = compute_confusion_matrix(val_preds, val_labels, num_classes)
        precision, recall, f1_score = compute_per_class_metrics(confusion_matrix)
        
        log_print(f"\n{'Class':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        log_print("-" * 60)
        
        for i in range(num_classes):
            support = confusion_matrix[i, :].sum()
            log_print(f"{i:<8} {precision[i]:<12.4f} {recall[i]:<12.4f} "
                     f"{f1_score[i]:<12.4f} {support:<10}")
        
        # Métricas macro e weighted
        macro_precision = precision.mean()
        macro_recall = recall.mean()
        macro_f1 = f1_score.mean()
        
        weights = confusion_matrix.sum(axis=1) / confusion_matrix.sum()
        weighted_precision = (precision * weights).sum()
        weighted_recall = (recall * weights).sum()
        weighted_f1 = (f1_score * weights).sum()
        
        log_print("")
        log_print(f"{'Macro Avg':<8} {macro_precision:<12.4f} {macro_recall:<12.4f} "
                 f"{macro_f1:<12.4f} {val_total:<10}")
        log_print(f"{'Weighted':<8} {weighted_precision:<12.4f} {weighted_recall:<12.4f} "
                 f"{weighted_f1:<12.4f} {val_total:<10}")
        log_print("")
        
        # Matriz de confusão (apenas primeiras 10 classes se houver muitas)
        log_print("-"*80)
        log_print("CONFUSION MATRIX (Validation Set)")
        log_print("-"*80)
        
        if num_classes <= 10:
            log_print("\nConfusion Matrix:")
            log_print("Rows = True class, Columns = Predicted class")
            log_print("")
            
            # Header
            header = "True\\Pred  " + "  ".join([f"{i:>6}" for i in range(num_classes)])
            log_print(header)
            log_print("-" * len(header))
            
            # Rows
            for i in range(num_classes):
                row = f"Class {i:>2}  " + "  ".join([f"{confusion_matrix[i, j]:>6}" for j in range(num_classes)])
                log_print(row)
        else:
            log_print("\nConfusion matrix is too large to display (>10 classes)")
            log_print(f"Diagonal sum (correct predictions): {np.trace(confusion_matrix)}")
            log_print(f"Off-diagonal sum (incorrect predictions): {confusion_matrix.sum() - np.trace(confusion_matrix)}")
        
        log_print("")
        log_print("="*80)
        log_print("END OF REPORT")
        log_print("="*80)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")
    
    return output_file

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test ULEEN model and generate report')
    parser.add_argument('--model', type=str, default='malevis_uleen_model.pt',
                       help='Path to the saved model file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file name (default: auto-generated with timestamp)')
    
    args = parser.parse_args()
    
    # Verificar se o arquivo do modelo existe
    if not os.path.exists(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        sys.exit(1)
    
    # Executar teste
    test_uleen_model(args.model, args.output)
