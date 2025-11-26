# 🛡️ Classificação de Malware usando WiSARD e Deep Learning

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![WiSARD](https://img.shields.io/badge/WiSARD-Weightless-green.svg)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange.svg)](https://www.tensorflow.org/)
[![Dataset](https://img.shields.io/badge/Dataset-MALEVIS-red.svg)]()

## 🎯 Sumário Executivo

Este repositório contém **17+ scripts Python** e **múltiplos notebooks** para classificação de malware usando:
- ✅ **WiSARD** (Redes Neurais sem Peso) - Rápido e eficiente
- ✅ **DenseNet201** (Deep Learning) - State-of-the-art accuracy
- ✅ **ClusWiSARD** (Clustering) - Supervisionado e não-supervisionado
- ✅ **Análise Visual** - Imagens 224x224 de binários PE
- ✅ **Análise Dinâmica** - PE imports (1000 features)

### 📊 Destaques
- **26 famílias de malware** agrupadas em **6 categorias**
- **8+ técnicas de codificação** (threshold, thermometer, LSB/MSB, unária)
- **85-95% acurácia** em classificação de vírus
- **Scripts especializados** por categoria (virus, trojan, worm, backdoor, adware)
- **Aprendizado incremental** com esquecimento seletivo

### 🚀 Quick Start
```bash
# 1. Gerar dados preprocessados
python gerarpickle.py

# 2. Primeiro experimento (rápido)
python main.py

# 3. Para produção
cd server
python wisard.py
```

## � Índice
1. [Descrição do Projeto](#-descrição-do-projeto)
2. [Dataset MALEVIS](#-dataset---malevis)
3. [Tecnologias Utilizadas](#-tecnologias-utilizadas)
4. [Estrutura dos Arquivos](#-estrutura-dos-arquivos)
5. [Resumo dos Arquivos por Categoria](#-resumo-dos-arquivos-por-categoria)
6. [Fluxo de Trabalho (Passo a Passo)](#-fluxo-de-trabalho-passo-a-passo)
7. [Técnicas de Preprocessamento](#-técnicas-de-preprocessamento)
8. [Algoritmos e Modelos](#-algoritmos-e-modelos)
9. [Guia de Uso - Qual Arquivo Escolher](#-guia-de-uso---qual-arquivo-escolher)
10. [Referência Rápida](#-referência-rápida---todos-os-arquivos-py)
11. [Configurações e Parâmetros](#️-configurações-e-parâmetros)
12. [Resultados Esperados](#-resultados-esperados)

## �📋 Descrição do Projeto

Este projeto implementa um sistema de classificação de malware utilizando técnicas de Machine Learning, especificamente redes neurais sem peso (WiSARD - Wilkie, Stonham and Aleksander Recognition Device) e redes neurais profundas (DenseNet). O sistema classifica malware em diferentes categorias usando visualizações de binários maliciosos.

## 🎯 Objetivo

Classificar arquivos maliciosos em diferentes categorias (26 classes específicas, 6 tipos gerais ou binário Malign/Benign) usando representações visuais de arquivos executáveis maliciosos (PE files).

## 📊 Dataset - MALEVIS

O projeto utiliza o dataset **MALEVIS** (Malware Visualization Dataset) que contém imagens de 224x224 pixels representando arquivos maliciosos. As imagens são geradas a partir da visualização de bytes de arquivos PE.

### Estrutura das Classes

O dataset é organizado em **26 famílias de malware** agrupadas em **6 categorias principais**:

1. **Adware** (6 famílias)
   - Adposhel, Amonetize, BrowseFox, InstallCore, MultiPlug, Neoreklami

2. **Trojan** (9 famílias)
   - Agent, Dinwod, Elex, HackKMS, Injector, Regrun, Snarasite, VBKrypt, Vilsel

3. **Worm** (4 famílias)
   - Allaple, Autorun, Fasong, Hlux

4. **Backdoor** (2 famílias)
   - Androm, Stantinko

5. **Virus** (4 famílias)
   - Expiro, Neshta, Sality, VBA

6. **Other/Benign**
   - Arquivos legítimos (goodware)

### Estrutura de Diretórios
```
malevis_train_val_224x224/
├── train/
│   ├── Adposhel/
│   ├── Agent/
│   ├── Allaple/
│   └── ... (26 pastas no total)
└── val/
    ├── Adposhel/
    ├── Agent/
    └── ... (26 pastas no total)
```

## 🔧 Tecnologias Utilizadas

- **Python 3.x**
- **WiSARD** (wisardpkg) - Rede neural sem peso
- **TensorFlow/Keras** - DenseNet201
- **OpenCV** - Processamento de imagens
- **scikit-learn** - Métricas e divisão de dados
- **NumPy/Pandas** - Manipulação de dados
- **Matplotlib** - Visualização

## 📁 Estrutura dos Arquivos

### Arquivos Principais

#### 1. **preprocess.py**
Preprocessamento e geração de arquivos pickle para uso posterior.
- Carrega imagens de todas as 26 classes
- Converte imagens RGB (224x224x3) para arrays unidimensionais
- Aplica threshold binário (valores > 127 → 1, caso contrário → 0)
- Salva dados preprocessados em arquivos pickle (X.p, y.p)

#### 2. **gerarpickle.py**
Gera arquivos pickle com diferentes níveis de granularidade de classificação:
- **y26.p**: 26 classes específicas (todas as famílias de malware)
- **y2.p**: Classificação binária (Malign vs Benign)
- **y6.p**: 6 categorias gerais (adware, Trojan, Worm, Backdoor, Virus, Benign)
- **data.p**: Dados de imagem brutos

#### 3. **main.py**
Script principal para treinamento e avaliação do modelo WiSARD com foco em classificação de vírus específicos.
- Carrega imagens das classes Expiro, Neshta e Sality
- Aplica preprocessamento dinâmico (threshold usando mediana)
- Treina modelo WiSARD com parâmetro de endereço configurável
- Avalia com métricas: F1-score, Precision, Recall, Accuracy
- Gera matriz de confusão

#### 4. **wisard.py**
Implementação completa de experimentos com WiSARD:
- Testa diferentes tamanhos de endereço (14 a 98 bits)
- Carrega dados de arquivos pickle
- Treina e testa modelo
- Gera métricas de performance e tempo de execução

#### 5. **virus.py**
Script avançado com múltiplas estratégias de classificação de vírus:

**Técnicas implementadas:**
- **Redimensionamento**: Testa imagens em diferentes resoluções
- **FFT**: Transformada rápida de Fourier para análise de frequência
- **Decision Trees com Voting**: Ensemble de árvores de decisão
- **PCA**: Redução de dimensionalidade antes da classificação
- **Mental Images**: Usa padrões aprendidos pelo WiSARD
- **WiSARD Pairs**: Treina classificadores binários para cada par de classes
- **WiSARD Voting**: Ensemble de WiSARDs com diferentes tamanhos de endereço (2, 4, 8, 16, 32, 64)
- **ClusWiSARD**: Versão de clustering da WiSARD (supervisionado e não supervisionado)

#### 6. **unary.py**
Implementação de codificação unária para representação de pixels:
- Converte valores de pixel (0-255) em representação unária de 256 bits
- Aumenta a dimensionalidade para capturar informações de intensidade

#### 7. **densenet.py**
Implementação de rede neural profunda usando DenseNet201:
- Transfer learning com DenseNet201
- Treinamento em 60 épocas com batch size de 64
- Logger customizado para acompanhar acurácia e perda

#### 8. **server/wisard.py**
Script otimizado para experimentos em lote:
- Suporta classificação em 3 níveis (2, 6 ou 26 classes)
- Usa codificação termômetro (8 níveis)
- Configuração via dicionário para experimentos parametrizados
- Salva resultados em arquivos de texto

#### 9. **server/forget.py**
Implementação de **aprendizado incremental com "esquecimento seletivo"**:
- Treina WiSARD de forma incremental (amostra por amostra)
- Para cada amostra de treino:
  - Treina o modelo com a amostra
  - Classifica a mesma amostra
  - Se errar a classificação, usa `leaveOneOut()` para "esquecer" o padrão incorreto
- Estratégia de correção de erros durante o treinamento
- Útil para lidar com ruído e overfitting
- Gera resultados em arquivo `thermometer8_[classe]forget.txt`

**Diferença chave**: Enquanto `wisard.py` faz treinamento em batch tradicional, `forget.py` implementa um mecanismo de autocorreção que remove padrões mal aprendidos.

#### 10. **test.py**
Script simples de teste para:
- Carregar e visualizar uma imagem de controle
- Converter de BGR para escala de cinza
- Testar funcionalidades básicas do OpenCV

#### 11. **main2.py**
Script de experimentação com **análise dinâmica de malware**:
- Usa dataset de **PE Imports** (top_1000_pe_imports.csv)
- Características extraídas de imports de arquivos PE (Windows executables)
- Classificação binária: malware vs goodware
- Balanceamento de classes com undersampling
- Suporta WiSARD e ClusWiSARD
- Opção de K-Fold Cross-Validation
- Foco em análise comportamental (não visual)

**Características:**
- Entrada: Vetor de features de imports (não imagens)
- 1000 features binárias representando imports de API do Windows
- Complementa análise visual com análise estática

#### 12. **wisard_sb.py** (Significant Bits)
Experimentos avançados com **separação de bits significativos**:
- **LSB (Least Significant Bits)**: 4 bits menos significativos
- **MSB (Most Significant Bits)**: 4 bits mais significativos
- Treina WiSARDs separados para LSB e MSB
- Combina resultados usando graus de ativação
- Extração de mental images do ClusWiSARD
- Dataset serialization (salva datasets em disco)
- Múltiplas estratégias: voting, pairs, activation degrees

**Técnica especial:**
```python
# Divide cada byte em MSB (4 bits) e LSB (4 bits)
for each pixel:
    binary = toBinary(pixel)
    lsb = bits[0:4]  # 4 menos significativos
    msb = bits[4:8]  # 4 mais significativos
```

#### 13. **server/main.py**
Script principal do servidor com suporte a **múltiplas codificações**:
- **Simple threshold**: Binarização fixa (threshold = 127)
- **Dynamic threshold**: Mediana adaptativa
- **Thermometer coding**: N níveis (4, 8, 12, etc.)
- **Circular thermometer**: Codificação circular com overlap
- Suporta classificação binária ou multiclasse (2, 26 classes)
- Interpolação opcional de imagens
- Processamento paralelo (multiprocessing)
- Configuração granular de experimentos

#### 14. **server/thermometer12.py**
Gerador de datasets com **codificação termômetro de 12 níveis**:
- Processa dados em blocos (0-999, 1000-1999, etc.)
- Cada pixel (0-255) → vetor de 12 bits
- Dimensão final: 150.528 → 1.806.336 bits
- Processamento eficiente para datasets grandes
- Salva blocos separados para economizar memória

**Codificação termômetro:**
```python
# Pixel value = 180 (exemplo)
# Range: 0-255 dividido em 12 bins
# Bin = floor(180 / (256/12)) = 8
# Resultado: [1,1,1,1,1,1,1,1,1,0,0,0]
```

#### 15. **server/join_dataset.py**
Utilitário para **unir blocos de datasets**:
- Concatena múltiplos arquivos pickle (X e y)
- Útil após processamento em blocos
- Cria dataset completo a partir de fragmentos
- Suporta qualquer tipo de codificação (thermometer8, thermometer12, etc.)

#### 16. **server/wisard_virus.py** (e wisard_adware.py, wisard_trojan.py, etc.)
Scripts especializados por **categoria de malware**:
- Cada arquivo foca em uma categoria específica
- Usa codificação thermometer12
- Filtra apenas as classes relevantes da categoria
- Configurações otimizadas por tipo de malware
- Permite experimentos focados e comparações entre categorias

**Categorias disponíveis:**
- `wisard_virus.py`: Expiro, Neshta, Sality, VBA
- `wisard_adware.py`: Adposhel, Amonetize, etc.
- `wisard_trojan.py`: Agent, Dinwod, Elex, etc.
- `wisard_worm.py`: Allaple, Autorun, Fasong, Hlux
- `wisard_backdoor.py`: Androm, Stantinko

#### 17. **server/wisard_6.py**
Classificação em **6 categorias principais**:
- adware, Trojan, Worm, Backdoor, Virus, Benign
- Agrupa as 26 famílias em tipos gerais
- Útil para classificação hierárquica
- Primeiro nível de granularidade

### Notebooks Jupyter

- **TF - Malware.ipynb**: Notebook principal com experimentos interativos
- **densenet.ipynb**: Experimentos com DenseNet
- **Untitled.ipynb / Untitled1.ipynb**: Notebooks de testes

## 🚀 Fluxo de Trabalho (Passo a Passo)

### Etapa 1: Preparação dos Dados

```bash
# 1. Organizar dataset MALEVIS na estrutura de pastas esperada
# malevis_train_val_224x224/train/ e malevis_train_val_224x224/val/

# 2. Gerar arquivos pickle para diferentes níveis de classificação
python gerarpickle.py
```

**O que acontece:**
- Carrega imagens RGB (224x224x3) de todas as 26 classes
- Cria três estruturas de labels: y26.p (26 classes), y2.p (binário), y6.p (6 categorias)
- Salva arrays NumPy em formato pickle para uso rápido

### Etapa 2: Preprocessamento (Opcional)

```bash
python preprocess.py
```

**O que acontece:**
- Converte imagens para arrays unidimensionais (224x224x3 → 150.528)
- Aplica threshold binário (127)
- Salva dados preprocessados em blocos para economizar memória

### Etapa 3: Treinamento com WiSARD

#### Opção A: Classificação de Vírus Específicos
```bash
python main.py
```

**Configuração:**
- Classes: Expiro, Neshta, Sality
- Preprocessamento: Threshold dinâmico (mediana)
- Address size: 20 bits
- Split: 70% treino, 30% teste

**Saída:**
- Métricas: F1-score, Precision, Recall, Accuracy
- Matriz de confusão

#### Opção B: Experimentos Completos
```bash
python wisard.py
```

**Configuração:**
- Usa dados preprocessados dos arquivos pickle
- Testa diferentes address sizes
- Avalia tempo de treinamento e inferência

#### Opção C: Técnicas Avançadas
```bash
python virus.py
```

**Opções disponíveis (configurar flags no código):**
- `wisard_voting = True`: Ensemble de WiSARDs
- `wisard_pairs = True`: Classificadores binários em pares
- `pca = True`: Redução de dimensionalidade + WiSARD
- `dt = True`: Decision Tree com mental images
- `mental_images = True`: Visualiza padrões aprendidos

### Etapa 4: Treinamento com Deep Learning

```bash
python densenet.py
```

**Configuração:**
- Arquitetura: DenseNet201
- Épocas: 60
- Batch size: 64
- Otimizador: Adam
- Loss: Categorical Crossentropy

### Etapa 5: Experimentos Parametrizados (Server)

```bash
cd server
python wisard.py
```

**Configuração:**
- Edite os parâmetros no topo do arquivo
- Escolha categoria: 'adware', 'Trojan', 'Worm', 'Backdoor', 'Virus'
- Define address size, número de runs, split size
- Resultados salvos em arquivo .txt

## � Resumo dos Arquivos por Categoria

### 🎯 Scripts Principais de Classificação
| Arquivo | Foco | Codificação | Classes |
|---------|------|-------------|---------|
| `main.py` | Vírus específicos | Mediana dinâmica | 3 vírus |
| `main2.py` | Análise dinâmica (PE imports) | Binário | 2 (malware/goodware) |
| `wisard.py` | Experimentos gerais | Threshold 127 | Configurável |
| `virus.py` | Técnicas avançadas | Mediana dinâmica | 3-4 vírus |
| `densenet.py` | Deep Learning | N/A (imagens RGB) | Configurável |

### 🔬 Scripts Experimentais Avançados
| Arquivo | Técnica Especial |
|---------|------------------|
| `wisard_sb.py` | Separação LSB/MSB (4+4 bits) |
| `unary.py` | Codificação unária (256 bits por pixel) |

### 🖥️ Scripts para Servidor (Produção)
| Arquivo | Função |
|---------|--------|
| `server/main.py` | Múltiplas codificações configuráveis |
| `server/wisard.py` | Thermometer 8 níveis |
| `server/wisard_6.py` | 6 categorias gerais |
| `server/wisard_[tipo].py` | Scripts especializados por categoria |
| `server/forget.py` | Aprendizado com esquecimento |

### 🛠️ Utilitários e Preprocessamento
| Arquivo | Função |
|---------|--------|
| `gerarpickle.py` | Gera pickles com 3 níveis de granularidade (2/6/26) |
| `preprocess.py` | Preprocessamento e threshold |
| `server/thermometer12.py` | Codificação termômetro 12 níveis em blocos |
| `server/join_dataset.py` | Une blocos de datasets |
| `test.py` | Testes básicos |

### 📊 Comparação de Abordagens

#### Por Tipo de Análise
1. **Análise Visual (Imagens)**: main.py, virus.py, wisard.py, densenet.py
2. **Análise Dinâmica (Imports)**: main2.py
3. **Análise Híbrida**: Combinar ambas abordagens

#### Por Granularidade de Classificação
1. **Binária (2 classes)**: Malign vs Benign
2. **Categórica (6 classes)**: adware, Trojan, Worm, Backdoor, Virus, Benign
3. **Específica (26 classes)**: Todas as famílias individuais
4. **Super específica (3-4 classes)**: Foco em vírus específicos

#### Por Tipo de Codificação
| Codificação | Dimensão Original | Dimensão Final | Arquivos |
|-------------|-------------------|----------------|----------|
| Threshold Binário | 150.528 | 150.528 | wisard.py, preprocess.py |
| Mediana Dinâmica | 150.528 | 150.528 | main.py, virus.py |
| Thermometer 4 | 150.528 | 602.112 | server/main.py |
| Thermometer 8 | 150.528 | 1.204.224 | server/wisard.py |
| Thermometer 12 | 150.528 | 1.806.336 | server/thermometer12.py |
| LSB/MSB (4+4) | 150.528 | 301.056 (cada) | wisard_sb.py |
| Unária | 150.528 | 38.535.168 | unary.py |
| Circular Thermometer | 150.528 | Configurável | server/main.py |

## �📊 Técnicas de Preprocessamento

### 1. **Threshold Binário**
```python
X = np.where(data > 127, 1, 0)
```
Converte pixels em valores binários.

### 2. **Threshold Dinâmico (Mediana)**
```python
for i in range(len(data)):
    med = np.median(data[i])
    data[i] = np.where(data[i] > med, 1, 0)
```
Usa a mediana como ponto de corte adaptativo.

### 3. **Codificação Termômetro (8 níveis)**
Divide o range 0-255 em 8 bins:
- Cada pixel gera um vetor de 8 bits
- Aumenta dimensão: 150.528 → 1.204.224

### 4. **Codificação Unária**
Converte cada pixel (0-255) em vetor de 256 bits:
- Dimensão final: 150.528 → 38.535.168

## 🔬 Algoritmos e Modelos

### WiSARD (Weightless Neural Network)

**Parâmetros principais:**
- **addressSize**: Tamanho do endereço de RAM (2-98 bits)
- **bleachingActivated**: Técnica para resolver empates
- **ignoreZero**: Ignora endereços 0 durante treinamento

**Vantagens:**
- Treinamento extremamente rápido (sem backpropagation)
- Inferência rápida
- Não requer GPU
- Boa interpretabilidade (mental images)

### ClusWiSARD (Clustering WiSARD)

Extensão da WiSARD para clustering:
- **Supervisionado**: Usa labels durante treinamento
- **Semi-supervisionado**: Usa alguns labels como seed
- **Não supervisionado**: Clustering puro

**Parâmetros:**
- minScore: Score mínimo para aceitar classificação
- threshold: Limiar de confiança
- discriminatorLimit: Número máximo de discriminadores

### DenseNet201

Rede neural convolucional profunda com conexões densas:
- 201 camadas
- Pré-treinamento disponível (ImageNet)
- Usado para transfer learning

## 📈 Métricas de Avaliação

Todas as implementações calculam:

- **Accuracy**: Taxa de acertos geral
- **Precision**: Precisão por classe (weighted average)
- **Recall**: Revocação por classe (weighted average)
- **F1-score**: Média harmônica entre precision e recall
- **Confusion Matrix**: Matriz de confusão normalizada
- **Training Time**: Tempo de treinamento
- **Testing Time**: Tempo de inferência

## 🎨 Visualizações

### Mental Images
```python
patterns = wsd.getMentalImages()
# Visualiza padrões aprendidos por cada discriminador
```

Gera imagens que representam o "conceito" aprendido de cada classe.

### Matriz de Confusão
```python
plot_confusion_matrix('confusion.png', y_test, y_pred, classes)
```

Visualização normalizada da performance do modelo.

## ⚙️ Configurações e Parâmetros

### Para Classificação de Vírus (main.py)

```python
# Classes a usar
classes = ["Expiro", "Neshta", "Sality"]

# Parâmetros WiSARD
addressSize = 20
bleachingActivated = True
ignoreZero = False

# Split de dados
SPLIT_SIZE = 0.3  # 30% teste, 70% treino
```

### Para Experimentos Server (server/wisard.py)

```python
# Escolher categoria
classes = 'Virus'  # ou 'adware', 'Trojan', 'Worm', 'Backdoor'

# Parâmetros
addressSize = 20
numberOfRuns = 1
SPLIT_SIZE = 0.3

# Tipo de modelo
wisard = True        # WiSARD padrão
cluswisard = False   # ClusWiSARD
```

## 📝 Arquivos de Saída

- **Matrices de Confusão**: `*.png`
- **Modelos Serializados**: `*.p` (pickle)
- **Resultados de Experimentos**: `thermometer8+[classe].txt`
- **Mental Images**: `expiro.png`, `neshta.png`, etc.
- **Decision Trees**: `decision.dot` (formato GraphViz)

## 🔍 Experimentos Interessantes Implementados

### 1. Ensemble de WiSARDs com Voting
Treina 6 WiSARDs com diferentes address sizes e usa votação majoritária.

### 2. WiSARD com Pares de Classes
Treina um classificador binário para cada par de classes (combinação 2 a 2).

### 3. PCA + WiSARD
Reduz dimensionalidade antes de treinar, testando diferentes números de componentes.

### 4. Decision Tree com Mental Images
Usa os padrões aprendidos por WiSARD como features para uma Decision Tree.

### 5. ClusWiSARD Não Supervisionado
Agrupa malware sem usar labels, depois mapeia clusters para classes reais.

### 6. Aprendizado Incremental com Esquecimento Seletivo (Forget)
Técnica implementada em `server/forget.py`:
- **Treinamento incremental**: Processa uma amostra por vez
- **Verificação imediata**: Classifica cada amostra logo após treinar
- **Correção de erros**: Se classificar incorretamente, usa `leaveOneOut()` para remover o padrão
- **Objetivo**: Prevenir que o modelo aprenda padrões incorretos ou ruidosos
- **Vantagem**: Reduz overfitting e melhora generalização

**Algoritmo:**
```python
for cada amostra de treino:
    treinar(amostra, label)
    predição = classificar(amostra)
    se predição != label:
        esquecer(amostra, label)  # leaveOneOut()
```

## 🚧 Estrutura do Código

```
TF - Malware/
├── README.md                    # Este arquivo
├── main.py                      # Script principal WiSARD (vírus)
├── main2.py                     # Análise dinâmica (PE imports)
├── wisard.py                    # Experimentos WiSARD completos
├── wisard_sb.py                 # WiSARD com Significant Bits (LSB/MSB)
├── virus.py                     # Técnicas avançadas (PCA, voting, etc.)
├── densenet.py                  # Deep Learning (DenseNet201)
├── preprocess.py                # Preprocessamento e geração de pickles
├── gerarpickle.py               # Geração de pickles multi-nível (2/6/26 classes)
├── unary.py                     # Codificação unária (256 bits)
├── test.py                      # Testes básicos OpenCV
├── *.p                          # Arquivos pickle (dados preprocessados)
├── TF - Malware.ipynb           # Notebook principal
├── densenet.ipynb               # Notebook DenseNet
├── Untitled*.ipynb              # Notebooks de experimentação
├── dynamic/                     # Dataset de análise dinâmica
│   └── top_1000_pe_imports.csv  # Features de PE imports
├── malevis_train_val_224x224/   # Dataset MALEVIS
│   ├── train/                   # Imagens de treino (26 pastas)
│   └── val/                     # Imagens de validação (26 pastas)
└── server/                      # Scripts otimizados para servidor
    ├── main.py                  # Múltiplas codificações (thermometer, circular, etc.)
    ├── wisard.py                # Experimentos parametrizados (thermometer8)
    ├── wisard_6.py              # Classificação 6 categorias
    ├── wisard_virus.py          # Específico para vírus
    ├── wisard_adware.py         # Específico para adware
    ├── wisard_trojan.py         # Específico para trojans
    ├── wisard_worm.py           # Específico para worms
    ├── wisard_backdoor.py       # Específico para backdoors
    ├── forget.py                # Aprendizado incremental com esquecimento
    ├── thermometer12.py         # Codificação termômetro 12 níveis
    ├── join_dataset.py          # União de blocos de datasets
    ├── gerarpickle.py           # Geração de pickles (versão servidor)
    ├── preproc.py               # Preprocessamento (versão servidor)
    └── dissertation/            # Scripts para dissertação
        ├── t.py                 # Experimentos termômetro
        ├── cr.py                # Experimentos circular
        ├── bt.py                # Binary threshold
        └── dbt.py               # Dynamic binary threshold
```

## 🎓 Referências e Conceitos

### WiSARD
- Rede neural sem peso baseada em RAMs
- Cada discriminador (classe) tem múltiplas RAMs
- Cada RAM mapeia um subconjunto de bits de entrada para 0 ou 1
- Classificação por contagem de RAMs ativadas

### Bleaching
Técnica para resolver empates:
1. Reduz gradualmente o threshold de aceitação
2. Reavaila até encontrar um vencedor claro

### Mental Images
Representação visual do que cada discriminador "aprendeu":
- Conta quantas vezes cada endereço foi acessado
- Normaliza e converte em imagem

### leaveOneOut()
Método especial do WiSARD para "desaprender" padrões:
- Remove o padrão específico das RAMs do discriminador
- Usado em `forget.py` para correção de erros
- Útil quando o modelo aprende algo incorretamente
- Diferente de retraining completo (é muito mais rápido)

## 🔄 Comparação: Treinamento Batch vs Incremental com Forget

### Treinamento Batch (wisard.py)
```python
wsd.train(X_train, y_train)  # Treina com todos os dados de uma vez
out = wsd.classify(X_test)    # Testa
```
**Vantagens:** Rápido, simples
**Desvantagens:** Pode memorizar ruído

### Treinamento Incremental com Forget (forget.py)
```python
for i in range(len(X_train)):
    wsd.train([X_train[i]], [y_train[i]])     # Treina uma amostra
    out = wsd.classify([X_train[i]])          # Verifica
    if out[0] != y_train[i]:                  # Se errou
        wsd.leaveOneOut([X_train[i]], [y_train[i]])  # Esquece
```
**Vantagens:** Mais robusto a ruído, autocorreção
**Desvantagens:** Mais lento, requer mais processamento

## 💡 Guia de Uso - Qual Arquivo Escolher?

### 🚀 Para Começar Rapidamente
```bash
python main.py
```
- **Quando usar**: Primeiro contato com o projeto, testes rápidos
- **Características**: 3 classes de vírus, código simples, resultados em minutos
- **Resultado esperado**: ~85-95% acurácia

### 🔬 Para Experimentação e Pesquisa

#### Técnicas Avançadas
```bash
python virus.py
```
- **Quando usar**: Testar diferentes estratégias (PCA, voting, pairs)
- **Características**: Múltiplas flags para ativar técnicas
- **Use para**: Comparar abordagens, encontrar melhor configuração

#### Significant Bits
```bash
python wisard_sb.py
```
- **Quando usar**: Analisar importância de bits MSB vs LSB
- **Características**: Treina modelos separados e combina
- **Use para**: Pesquisa sobre representação de features

#### Análise Dinâmica
```bash
python main2.py
```
- **Quando usar**: Análise comportamental (não visual)
- **Características**: Usa imports de PE files
- **Use para**: Complementar análise de imagens

### 🖥️ Para Produção e Experimentos em Larga Escala

#### Experimentos Parametrizados
```bash
cd server
python wisard.py  # Thermometer 8
```
- **Quando usar**: Múltiplos runs, resultados estatísticos
- **Características**: Configurável, salva resultados em arquivo
- **Use para**: Benchmarks, comparações formais

#### Por Categoria Específica
```bash
python server/wisard_virus.py
python server/wisard_adware.py
python server/wisard_trojan.py
```
- **Quando usar**: Análise focada em um tipo de malware
- **Características**: Otimizado para cada categoria
- **Use para**: Estudos especializados, classificadores dedicados

#### Com Esquecimento Seletivo
```bash
python server/forget.py
```
- **Quando usar**: Dataset com ruído, necessidade de robustez
- **Características**: Autocorreção durante treinamento
- **Use para**: Melhorar generalização, lidar com outliers

#### Múltiplas Codificações
```bash
python server/main.py
```
- **Quando usar**: Testar diferentes codificações (thermometer, circular)
- **Características**: Suporte a 5+ tipos de codificação
- **Use para**: Encontrar melhor representação para seus dados

### 🤖 Para Deep Learning
```bash
python densenet.py
```
- **Quando usar**: Buscar state-of-the-art accuracy
- **Características**: DenseNet201, transfer learning
- **Requisitos**: GPU recomendada, mais tempo de treinamento
- **Use para**: Baseline de comparação, produção

### 🛠️ Para Preprocessamento

#### Gerar Pickles
```bash
python gerarpickle.py  # Gera 3 níveis: 2/6/26 classes
```
- **Quando usar**: Primeira vez, ou mudança no dataset
- **Resultado**: data.p, y2.p, y6.p, y26.p

#### Thermometer em Blocos
```bash
python server/thermometer12.py  # Processa em blocos
python server/join_dataset.py   # Une os blocos
```
- **Quando usar**: Dataset grande, memória limitada
- **Resultado**: thermometer12_X.p, thermometer12_y.p

## 📊 Fluxo de Trabalho Recomendado

### Iniciante
1. `gerarpickle.py` → Gerar dados
2. `main.py` → Primeiro experimento
3. `virus.py` → Explorar técnicas

### Pesquisador
1. `gerarpickle.py` → Preparar dados
2. `server/thermometer12.py` + `join_dataset.py` → Codificação avançada
3. `server/wisard.py` → Benchmarks
4. `virus.py` → Técnicas específicas
5. `densenet.py` → Comparação com DL

### Produção
1. `server/thermometer12.py` → Preparação
2. `server/wisard_[categoria].py` → Modelos especializados
3. `server/forget.py` → Versão robusta
4. Deploy do melhor modelo

## 💡 Dicas de Uso

1. **Para resultados rápidos**: Use `main.py` com 3 classes de vírus
2. **Para experimentação**: Use `virus.py` e ative diferentes flags
3. **Para produção**: Use `server/wisard.py` com múltiplos runs
4. **Para deep learning**: Use `densenet.py` (requer GPU recomendado)
5. **Para análise comportamental**: Use `main2.py` com PE imports
6. **Para robustez**: Use `server/forget.py` com esquecimento seletivo
7. **Para eficiência**: Use `server/thermometer12.py` em blocos
8. **Para especialização**: Use `server/wisard_[tipo].py` por categoria

## ⚠️ Requisitos de Sistema

- **RAM**: Mínimo 8GB (16GB recomendado para codificação termômetro)
- **Armazenamento**: ~5GB para dataset + arquivos pickle
- **GPU**: Opcional, mas recomendado para DenseNet
- **CPU**: Multi-core recomendado para WiSARD voting

## 📦 Dependências

```bash
pip install numpy pandas matplotlib opencv-python scikit-learn
pip install wisardpkg tensorflow keras
pip install imblearn  # Para balanceamento de classes (opcional)
```

## 🎯 Resultados Esperados

- **WiSARD (3 classes)**: ~85-95% acurácia
- **WiSARD (26 classes)**: ~70-80% acurácia
- **WiSARD Voting**: +5-10% sobre WiSARD simples
- **DenseNet201**: ~90-95% acurácia (26 classes)

## � Referência Rápida - Todos os Arquivos .py

### Raiz do Projeto

| Arquivo | Linha de Comando | Propósito | Tempo Estimado |
|---------|------------------|-----------|----------------|
| `main.py` | `python main.py` | Classificação de 3 vírus (Expiro, Neshta, Sality) | ~5-10 min |
| `main2.py` | `python main2.py` | Análise dinâmica com PE imports | ~10-20 min |
| `wisard.py` | `python wisard.py` | Experimentos com pickles pré-processados | ~15-30 min |
| `wisard_sb.py` | `python wisard_sb.py` | Separação LSB/MSB, mental images | ~20-40 min |
| `virus.py` | `python virus.py` | Técnicas avançadas (ativar flags) | ~10-60 min |
| `densenet.py` | `python densenet.py` | Deep learning com DenseNet201 | ~2-4 horas |
| `unary.py` | `python unary.py` | Codificação unária (256 bits) | ~30-60 min |
| `preprocess.py` | `python preprocess.py` | Gera X.p e y.p com threshold | ~20-40 min |
| `gerarpickle.py` | `python gerarpickle.py` | Gera data.p, y2.p, y6.p, y26.p | ~10-20 min |
| `test.py` | `python test.py` | Teste básico OpenCV | ~1 seg |

### Pasta server/

| Arquivo | Linha de Comando | Propósito | Tempo Estimado |
|---------|------------------|-----------|----------------|
| `main.py` | `cd server && python main.py` | Múltiplas codificações configuráveis | ~30-60 min |
| `wisard.py` | `cd server && python wisard.py` | Thermometer 8, categoria específica | ~15-30 min |
| `wisard_6.py` | `cd server && python wisard_6.py` | Classificação em 6 categorias | ~20-40 min |
| `wisard_virus.py` | `cd server && python wisard_virus.py` | Específico para vírus (Thermometer 12) | ~15-25 min |
| `wisard_adware.py` | `cd server && python wisard_adware.py` | Específico para adware | ~15-25 min |
| `wisard_trojan.py` | `cd server && python wisard_trojan.py` | Específico para trojans | ~15-25 min |
| `wisard_worm.py` | `cd server && python wisard_worm.py` | Específico para worms | ~10-15 min |
| `wisard_backdoor.py` | `cd server && python wisard_backdoor.py` | Específico para backdoors | ~8-12 min |
| `forget.py` | `cd server && python forget.py` | Aprendizado com esquecimento | ~20-40 min |
| `thermometer12.py` | `cd server && python thermometer12.py` | Gera blocos thermometer 12 | ~5-10 min/bloco |
| `join_dataset.py` | `cd server && python join_dataset.py` | Une blocos de datasets | ~2-5 min |
| `gerarpickle.py` | `cd server && python gerarpickle.py` | Versão servidor (mesmo que raiz) | ~10-20 min |
| `preproc.py` | `cd server && python preproc.py` | Versão servidor preprocessamento | ~20-40 min |

### Pasta server/dissertation/

| Arquivo | Propósito |
|---------|-----------|
| `t.py` | Experimentos com thermometer (dissertação) |
| `cr.py` | Experimentos com circular thermometer |
| `bt.py` | Binary threshold experiments |
| `dbt.py` | Dynamic binary threshold experiments |

## 🎓 Glossário de Termos

- **WiSARD**: Rede neural sem peso (Weightless)
- **Bleaching**: Técnica de desempate
- **Mental Images**: Visualização de padrões aprendidos
- **Address Size**: Tamanho do endereço de RAM (bits)
- **Thermometer Coding**: Codificação onde N bins geram N bits
- **LSB/MSB**: Least/Most Significant Bits
- **PE Imports**: Imports de API do Windows em executáveis
- **ClusWiSARD**: Versão clustering da WiSARD
- **Leave One Out**: Método para "esquecer" padrões

## �🔬 Trabalhos Futuros

- Implementar outras codificações (Gray, One-Hot)
- Testar com outros datasets de malware
- Implementar análise dinâmica de comportamento
- Criar API REST para classificação em tempo real
- Otimizar para edge devices
- Combinar análise visual + comportamental
- Implementar detecção de zero-day malware
- Transfer learning com outras CNNs (ResNet, EfficientNet)

## � Estatísticas do Projeto

### Arquivos Python
- **Total**: 17+ arquivos .py
- **Raiz**: 10 arquivos principais
- **Server**: 13+ arquivos especializados
- **Dissertation**: 4 arquivos de experimentos

### Técnicas Implementadas
- ✅ 8+ tipos de codificação de features
- ✅ 5 estratégias de ensemble (voting, pairs, etc.)
- ✅ 3 níveis de granularidade (2/6/26 classes)
- ✅ 2 tipos de análise (visual + dinâmica)
- ✅ Aprendizado incremental com forget
- ✅ PCA, Decision Trees, ClusWiSARD
- ✅ Mental images e visualização

### Dataset
- **Imagens**: 9.100 amostras (224x224x3)
- **Classes**: 26 famílias + 1 benign
- **Categorias**: 6 tipos de malware
- **PE Imports**: 1.000 features dinâmicas

### Performance
- **WiSARD (3 classes)**: 85-95% acurácia
- **WiSARD (26 classes)**: 70-80% acurácia  
- **DenseNet**: 90-95% acurácia
- **Tempo treino WiSARD**: Segundos a minutos
- **Tempo treino DenseNet**: 2-4 horas

## 🤝 Contribuindo

Sugestões de melhorias:
1. Adicionar mais técnicas de codificação
2. Implementar outras arquiteturas de DL
3. Criar pipeline automatizado
4. Desenvolver interface web/API
5. Otimizar para deployment

## 📚 Referências

- **WiSARD**: Aleksander et al., "WISARD: A Radical Step Forward in Image Recognition"
- **MALEVIS**: Malware Visualization Dataset
- **DenseNet**: Huang et al., "Densely Connected Convolutional Networks"
- **Malware Analysis**: PE file structure and static analysis

## �📧 Autor

Pedro Henrique Teixeira

## 📄 Licença

Este projeto é disponibilizado para fins acadêmicos e de pesquisa.

---

**⚠️ Nota Importante**: Este é um projeto de pesquisa para classificação de malware. Os modelos devem ser usados como parte de um sistema de segurança multicamadas, não como única defesa. Sempre use em conjunto com antivírus comerciais e outras técnicas de detecção.

---

**🌟 Se este projeto foi útil, considere dar uma estrela no repositório!**
