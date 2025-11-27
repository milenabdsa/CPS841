# bloomwisard_6.py
# Classificação das 6 famílias usando BloomWiSARD em cima do mesmo dataset
# que o wisard_6.py usa (thermometer12 + labels y6).

from pathlib import Path
import pickle
import numpy as np

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

import torch
from torchwnn.classifiers import BloomWiSARD


# ==========================
# CONFIGURAÇÕES GERAIS
# ==========================
# limita o número de amostras pra não matar a máquina
MAX_TRAIN_SAMPLES = 3000      # ajusta se ficar pesado
MAX_VAL_SAMPLES = 1000        # idem
TUPLE_SIZE = 20               # equivalente ao addressSize / tupleSize
RANDOM_STATE = 42

# Mapeamento de labels (mesmo espírito do y6)
CLASS_NAMES = [
    "Adware",
    "Trojan",
    "Worm",
    "Backdoor",
    "Virus",
    "Benign"
]


# ==========================
# FUNÇÕES AUXILIARES
# ==========================

def resolve_paths():
    """
    Resolve caminhos de forma robusta:
    BASE_DIR = raiz do projeto (onde ficam p/ e src/)
    """
    this_file = Path(__file__).resolve()
    src_dir = this_file.parent             # .../py/src
    base_dir = src_dir.parent              # .../py
    p_dir = base_dir / "p"
    out_dir = p_dir / "bloomwisard"

    out_dir.mkdir(parents=True, exist_ok=True)

    return base_dir, p_dir, out_dir


def load_data(p_dir):
    """
    Carrega:
      - thermometer12_train_X.p
      - thermometer12_val_X.p
      - y6.p
      - y6_val.p
    converte pra tensores do PyTorch.
    """

    with open(p_dir / "thermometer12_train_X.p", "rb") as f:
        X_train = pickle.load(f)

    with open(p_dir / "thermometer12_val_X.p", "rb") as f:
        X_val = pickle.load(f)

    with open(p_dir / "y6.p", "rb") as f:
        y_train = pickle.load(f)

    with open(p_dir / "y6_val.p", "rb") as f:
        y_val = pickle.load(f)

    # converte pra numpy pra garantir shape bonitinho
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    y_train = np.array(y_train).astype(int).ravel()
    y_val = np.array(y_val).astype(int).ravel()

    return X_train, X_val, y_train, y_val


def subsample(X, y, max_samples):
    """
    Faz um corte estratificado pra não passar do limite de amostras.
    """
    if max_samples is None or len(X) <= max_samples:
        return X, y

    from sklearn.model_selection import train_test_split

    _, X_sub, _, y_sub = train_test_split(
        X, y,
        test_size=max_samples,
        stratify=y,
        random_state=RANDOM_STATE
    )
    return X_sub, y_sub


def to_torch(X, y, device):
    """
    Converte numpy -> tensores do PyTorch.
    """
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    return X_t, y_t


def train_and_eval_bloomwisard():
    base_dir, p_dir, out_dir = resolve_paths()

    print(f"BASE_DIR = {base_dir}")
    print(f"Carregando dados em {p_dir} ...")

    X_train, X_val, y_train, y_val = load_data(p_dir)

    print(f"Tamanho original do treino: {len(X_train)}")
    print(f"Tamanho original do val   : {len(X_val)}")

    X_train, y_train = subsample(X_train, y_train, MAX_TRAIN_SAMPLES)
    X_val, y_val = subsample(X_val, y_val, MAX_VAL_SAMPLES)

    print(f"Tamanho usado no treino: {len(X_train)}")
    print(f"Tamanho usado no val   : {len(X_val)}")

    # Flatten se ainda não estiver flatten
    X_train = X_train.reshape(len(X_train), -1)
    X_val = X_val.reshape(len(X_val), -1)

    entry_size = X_train.shape[1]
    n_classes = len(np.unique(y_train))

    print(f"Dimensão de entrada (entry_size): {entry_size}")
    print(f"Número de classes: {n_classes}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")

    X_train_t, y_train_t = to_torch(X_train, y_train, device)
    X_val_t, y_val_t = to_torch(X_val, y_val, device)

    # ==========================
    # CRIA E TREINA BloomWiSARD
    # =================