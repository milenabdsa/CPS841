# Projeto MALWISARD 
## Pipeline de Execução

Esse README descreve o passo a passo para reproduzir o pipeline que foi testado no projeto `projeto_malwisard`, incluindo:
- Geração dos arquivos de pré-processamento (`data.p`, `y2.p`, `y6.p`, `y26.p`)
- Binarização com Thermometer Encoding (N=12)
- Junção dos “chunks” em um único dataset binarizado
- Execução dos classificadores WiSARD por família de malware
- Execução dos experimentos específicos com vírus (`virus.py`)

As instruções abaixo (você deve estar na raiz do projeto)
```bash
cd projeto_malwisard
```

Estrutura esperada do projeto
Na raiz de `projeto_malwisard/` você deve ter, no mínimo:
- `malevis_train_val_224x224/` – imagens do dataset MALICIOUS / GOODWARE
  - `train/<classe>/...` – pastas com as 26 classes de malware/goodware
- `src/` – scripts Python do pipeline
  - `gerarpickle.py`
  - `thermometer12.py`
  - `join_dataset.py`
  - `wisard_adware.py`
  - `wisard_trojan.py`
  - `wisard_worm.py`
  - `wisard_backdoor.py`
  - `wisard_virus.py`
  - `virus.py`
- (Demais arquivos gerados: `data.p`, `y*.p`, `thermometer12_*.p`, etc.)


## Passo 1: Pré-processamento com `src/gerarpickle.py` ✅
**Objetivo:** ler todas as imagens do dataset e gerar os arquivos base de dados e rótulos.
**Script:** `src/gerarpickle.py`
**Entrada:** imagens em `malevis_train_val_224x224/train/<classe>/...`
**Processamento:**
  - Lê as imagens RGB 224×224×3 de todas as 26 classes.
  - Achata as imagens em vetores (`224*224*3`) e monta um grande array.
  - Cria diferentes vetores de rótulos:
    - `y26.p` – rótulo com as **26 classes** originais.
    - `y2.p` – rótulo **binário** (Goodware vs Malign).
    - `y6.p` – rótulos **agrupados** em 6 categorias (Adware, Trojan, Worm, Backdoor, Virus, Benign/Other).

### Arquivos gerados (na raiz de `projeto_malwisard/`)
- `data.p` — ~1519.1 MB (dados das imagens processadas)
- `y26.p` — ~0.44 MB (labels com 26 classes)
- `y2.p`  — ~0.24 MB (labels binários: Malign/Benign)
- `y6.p`  — ~0.32 MB (labels com 6 classes)
O script processa todas as 26 classes de malware do dataset.

### Como executar
Na raiz de `projeto_malwisard/`:
```bash
python src/gerarpickle.py
```

Após rodar, você deve ver na raiz:
- `data.p`
- `y26.p`
- `y2.p`
- `y6.p`

## Passo 2: Binarização com thermometer encoding (`src/thermometer12.py`) ✅

**Objetivo:** transformar os vetores contínuos em representações binárias usando Thermometer Encoding (N=12).
**Script:** `src/thermometer12.py`
**Entrada:** `data.p` e `y26.p` (gerados no Passo 1)
**Parâmetros principais (no topo do arquivo):**
  - `inicio` – índice inicial das amostras (ex.: `0`)
  - `fim` – índice final (ex.: `1000`)
  - `filename` – prefixo dos arquivos de saída (ex.: `'thermometer12_0_999'`)
**Processamento:**
  - Recorta o intervalo `[inicio:fim]` do dataset (`data.p` e `y26.p`).
  - Aplica Thermometer Encoding com `N = 12` bits.
  - Salva os “pedaços” (chunks) em arquivos `.p`.

### Arquivos gerados (exemplo)
Para `inicio = 0`, `fim = 1000`, `filename = 'thermometer12_0_999'`, serão gerados:
- `thermometer12_0_999X.p` – ~1149.75 MB (dados binarizados)
- `thermometer12_0_999y.p` – ~0.04 MB (labels correspondentes)
Esses arquivos ficam na raiz de `projeto_malwisard/` (ou na pasta onde o script foi configurado para salvar).

### Estratégia original em blocos
Para seguir a lógica do autor (dataset completo), o fluxo seria rodar várias vezes o script, ajustando `inicio`, `fim` e `filename`:
- `inicio = 0`,   `fim = 1000`, `filename = 'thermometer12_0_999'`
- `inicio = 1000`, `fim = 2000`, `filename = 'thermometer12_1000_1999'`
- `inicio = 2000`, `fim = 3000`, `filename = 'thermometer12_2000_2999'`
- ...
- `inicio = 8000`, `fim = 9000`, `filename = 'thermometer12_8000_8999'`
- `inicio = 9000`, `fim = 9100`, `filename = 'thermometer12_9000_9099'` (configuração original do autor)
Cada execução gera um par `X` / `y` para aquele intervalo.

### Cenário de testes
Nos testes realizados aqui, foi gerado apenas o bloco:
- `inicio = 0`
- `fim = 1000`
- `filename = 'thermometer12_0_999'`
E o arquivo foi deixado de volta com o valor original (`inicio = 9000`, `fim = 9100`) no código, para manter a referência do autor.

### Como executar
Editar o topo de `src/thermometer12.py` com:
```python
inicio = 0
fim = 1000
filename = 'thermometer12_0_999'
```

Depois, na raiz de `projeto_malwisard/`:
```bash
python src/thermometer12.py
```
Repetir o processo alterando `inicio`, `fim` e `filename` para gerar os outros blocos

## Passo 3: Junção dos chunks em um único dataset (`src/join_dataset.py`) ✅
**Objetivo:** juntar todos os chunks `thermometer12_*X.p` e `thermometer12_*y.p` em um único dataset binarizado.
**Script:** `src/join_dataset.py`
**Versão original:** esperava os arquivos:
  - `thermometer12_0_999X.p`, `thermometer12_1000_1999X.p`, ..., `thermometer12_9000_9099X.p`
  - e os correspondentes `y`
**Saída esperada:**
  - `thermometer12_X.p`
  - `thermometer12_y.p`

### Problema com uso parcial (apenas 0–999)
Se você gera apenas o bloco `0–999`, terá só:
- `thermometer12_0_999X.p`
- `thermometer12_0_999y.p`

Se rodar a versão original de `join_dataset.py`, ela tenta carregar:
- `thermometer12_0_999X.p`
- `thermometer12_1000_1999X.p`
- `thermometer12_2000_2999X.p`
- ...
- `thermometer12_9000_9099X.p`
e vai falhar com `FileNotFoundError` assim que chegar em um arquivo que não existe (por exemplo, `2000_2999X.p`).

### Estratégia usada nos testes (subset 0–999)
Para demonstrar o pipeline sem “matar” a máquina, foi usado apenas o bloco 0–999:
1. Rodado o `thermometer12.py` apenas para `inicio = 0`, `fim = 1000`.
2. Gerados:
   - `thermometer12_0_999X.p`
   - `thermometer12_0_999y.p`
3. Em vez de usar o join completo, foi feito o “atalho”:

```bash
mv thermometer12_0_999X.p thermometer12_X.p
mv thermometer12_0_999y.p thermometer12_y.p
```
Assim, todos os scripts que esperam `thermometer12_X.p` e `thermometer12_y.p` funcionam com esse subset de 1000 amostras.

### Versão adaptada do `join_dataset.py`
O `join_dataset.py` foi ajustado para:
- Procurar automaticamente os arquivos `thermometer12_*X.p` e `thermometer12_*y.p` que realmente existem na mesma pasta.
- Carregar apenas esses chunks.
- Concatenar em:
  - `thermometer12_X.p`
  - `thermometer12_y.p`
Uso recomendado:
1. Certifique-se de que na mesma pasta de `join_dataset.py` (tipicamente `src/`) existam:
   - `join_dataset.py`
   - `thermometer12_0_999X.p`
   - `thermometer12_0_999y.p`
   - (ou outros `thermometer12_*X.p` / `thermometer12_*y.p` que você tenha gerado).
2. No terminal:

```bash
cd projeto_malwisard/src
python join_dataset.py
```

Saída esperada:
- Mensagens do tipo:
  - “Procurando arquivos de chunks thermometer12_*X.p ...”
  - “Carregando thermometer12_0_999X.p e thermometer12_0_999y.p...”
  - “Total de chunks usados: 1000”
  - “Total de rótulos: 1000”
  - “Salvando dataset final em thermometer12_X.p e thermometer12_y.p ...”
  - “Concluído!”
- Arquivos gerados na mesma pasta:
  - `thermometer12_X.p`
  - `thermometer12_y.p`
Esses dois arquivos serão usados pelos scripts WiSARD e pelo `virus.py` nos próximos passos.


## Passo 4: Classificadores WiSARD por família (`src/wisard_*.py`) 
**Objetivo:** reproduzir as classificações por família (Adware, Trojan, Worm, Backdoor, Virus) como descritas no artigo.
**Scripts principais:**
- `src/wisard_adware.py`   
- `src/wisard_trojan.py`   
- `src/wisard_worm.py`     
- `src/wisard_backdoor.py` (falhou no subset 0–999)
- `src/wisard_virus.py`    (falhou no subset 0–999)

Todos esses scripts:
- Carregam `thermometer12_X.p` e `thermometer12_y.p`.
- Filtram apenas as classes da família em questão.
- Fazem split em treino/teste (por `SPLIT_SIZE` / proporção definida no código).
- Treinam uma rede WiSARD.
- Imprimem métricas: acurácia, precisão, recall, F1.
- Em alguns casos, geram também matriz de confusão e gravam arquivos `thermometer12_*.txt` com resultados.

### Como executar

Na raiz de `projeto_malwisard/` (garantindo que `thermometer12_X.p` e `thermometer12_y.p` já existem):
```bash
python src/wisard_adware.py
python src/wisard_trojan.py
python src/wisard_worm.py
python src/wisard_backdoor.py
python src/wisard_virus.py
```

### Resultados obtidos com o subset 0–999
Com `thermometer12_X.p` e `thermometer12_y.p` contendo apenas as 1000 primeiras amostras (0–999), a distribuição de classes foi:
- Adposhel (Adware): 350 amostras
- Agent (Trojan): 350 amostras
- Allaple (Worm): 300 amostras
- Backdoor: 0 amostras
- Virus: 0 amostras

Resultados:
- `wisard_adware.py`
  - Acurácia: 100% (1.0)
  - F1-score: 1.0
  - Precision: 1.0
  - Recall: 1.0
  - Tempo de treino: ~3.30s
  - Tempo de teste: ~1.41s
- `wisard_trojan.py`
  - Acurácia: 100% (1.0)
  - F1-score: 1.0
  - Precision: 1.0
  - Recall: 1.0
  - Tempo de treino: ~4.03s
  - Tempo de teste: ~1.42s
- `wisard_worm.py`
  - Acurácia: 100% (1.0)
  - F1-score: 1.0
  - Precision: 1.0
  - Recall: 1.0
  - Tempo de treino: ~3.42s
  - Tempo de teste: ~1.19s
- `wisard_backdoor.py`
  - Falha: `ValueError` / `n_samples=0`
  - Motivo: o subset 0–999 não contém amostras de Backdoor.
- `wisard_virus.py`
  - Falha: `ValueError` / `n_samples=0`
  - Motivo: o subset 0–999 não contém amostras de Virus.

Ou seja, os classificadores que foram executados (Adware, Trojan, Worm) alcançaram 100% de acurácia neste subset reduzido, pois o dataset contém apenas três classes. Para testar Backdoor e Virus adequadamente e reproduzir fielmente os resultados do artigo, é necessário:
- Gerar mais intervalos com `thermometer12.py` (por exemplo, até cobrir todo o dataset original), ou
- Usar um subset maior que inclua amostras dessas classes.

Os resultados e métricas são salvos em arquivos `src/thermometer12_*.txt`.


## Passo 5: Experimentos específicos com vírus (`src/virus.py`) ✅

**Objetivo:** executar os experimentos específicos com vírus, incluindo:
- Simple WiSARD
- Ensemble
- Mental image
- Decision Tree sobre mental images
**Script principal:** `src/virus.py`

### Ajuste necessário (`import gist`)
O script original continha a linha:
```python
import gist
```
No ambiente atual, o módulo `gist` não existe e não é utilizado pelo restante do código. Isso gerava:
- `ModuleNotFoundError: No module named 'gist'`
Solução aplicada:
- Remover a linha `import gist` de `src/virus.py`.
- Após essa remoção, o script funciona corretamente.

### Pré-requisitos
- Estrutura de pastas:
```text
projeto_malwisard/
  malevis_train_val_224x224/
  src/
    virus.py
    ...
  thermometer12_X.p
  thermometer12_y.p
```
- `thermometer12_X.p` e `thermometer12_y.p` já gerados (via `join_dataset.py` ou renomeando os arquivos do passo 2).

### Como executar
Na raiz de `projeto_malwisard/`:
```bash
python src/virus.py
```

### Resultado observado (modo “dt” – Decision Tree)
No experimento executado:
- Carregou imagens das classes: Expiro, Neshta e Sality.
- Treinou a rede WiSARD.
- Extraiu mental images.
- Treinou uma árvore de decisão (`Decision Tree`) usando as mental images.
- Testou o modelo e obteve:
  - Acurácia: ~0.3397 (33.97%).
- Gerou o arquivo:
  - `decision.dot` – árvore de decisão em formato Graphviz.
O script funciona corretamente quando executado a partir da pasta raiz `projeto_malwisard/` com a estrutura acima.

---

## Passo 6: Classificador WiSARD com 6 classes agregadas (`src/wisard_6.py`) ✅

No passo 6, o script `wisard_6.py` treina uma única WiSARD para 6 classes agregadas (Adware, Trojan, Worm, Backdoor, Virus, Benign/Other) usando vetores muito grandes. Rodar com todas as amostras de `thermometer12_X.p` pode ser extremamente pesado, então foram feitas adaptações para deixar o experimento mais leve, mantendo o pipeline fiel ao autor.

### Limitar o número máximo de amostras (`MAX_SAMPLES`)
Logo no início do arquivo, após:
```python
SPLIT_SIZE = 0.3
numberOfRuns = 1
addressSize = 20
```
foi adicionado:
```python
MAX_SAMPLES = 500  
```
Em máquinas mais fortes, você pode testar `MAX_SAMPLES = 1000`, mas nos testes atuais foi necessário usar 500 para rodar sem erro/memória estourando.

### Cortar o dataset para no máximo `MAX_SAMPLES`

Depois do trecho:
```python
print("Importing data...")

with open("thermometer12_X.p", "rb") as input_file:
  X = pd.read_pickle(input_file, compression=None)

print("X imported")

with open("thermometer12_y.p", "rb") as input_file:
  y = pd.read_pickle(input_file, compression=None)

print("y imported")

gc.collect()
```
foi incluído o corte de amostras:
```python
print("Dataset completo: ", len(X), "amostras")
if len(X) > MAX_SAMPLES:
    from sklearn.model_selection import train_test_split
    print(f"Reduzindo para {MAX_SAMPLES} amostras para acelerar o experimento...")
    _, X_sample, _, y_sample = model_selection.train_test_split(
        X, y,
        test_size=MAX_SAMPLES,
        stratify=y,
        random_state=42
    )
    X = X_sample
    y = y_sample
    print("Novo tamanho do dataset:", len(X))
else:
    print("Mantendo todas as amostras.")
```
Assim, mesmo que `thermometer12_X.p` tenha 10k+ amostras, o `wisard_6.py` só usa no máximo `MAX_SAMPLES`, o que torna a execução viável para demonstração.

### Desligar o `verbose` da WiSARD

O código original instanciava a WiSARD com `verbose=True`, deixando o treino lento e floodando o terminal:
```python
if wisard:
    print("WiSARD")
    wsd = wp.Wisard(addressSize, bleachingActivated = bleachingActivated, ignoreZero = ignoreZero, verbose = True)
```
Foi alterado para:
```python
if wisard:
    print("WiSARD")
    wsd = wp.Wisard(addressSize, bleachingActivated=bleachingActivated, ignoreZero=ignoreZero, verbose=False)
```
Com isso, o modelo treina muito mais rápido e sem mensagens desnecessárias.

### 6.4 – Execução e resultados obtidos

Na raiz de `projeto_malwisard/` (com `thermometer12_X.p` e `thermometer12_y.p` já gerados), execute:
```bash
python src/wisard_6.py
```
Resultados observados nos testes (com limite de amostras aplicado):
- Acurácia: 90.67% (0.9067)
- F1-score: 90.67% (0.9067)
- Precision: 91.38% (0.9138)
- Recall: 90.67% (0.9067)
- Tamanho do treino: 350 amostras
- Tamanho do teste: 150 amostras
- Tempo de treino: ~6.03s
- Tempo de teste: ~5.13s
Arquivos gerados:
- `thermometer12_6classes.txt` – resultados do experimento
- `thermometer12_6classes.png` – matriz de confusão (se estiver habilitada no script)

## Resumo rápido do fluxo completo

1. Pré-processamento (todas as imagens):
   ```bash
   cd projeto_malwisard
   python src/gerarpickle.py
   ```

2. Binarização Thermometer (por blocos, ex. 0–999):
   - Editar `inicio`, `fim`, `filename` em `src/thermometer12.py`.
   - Rodar:
   ```bash
   python src/thermometer12.py
   ```

3. Join dos chunks (ou renomear 0–999 como dataset completo de teste):
   - Via join automático:
   ```bash
   cd projeto_malwisard/src
   python join_dataset.py
   ```
   - Ou, para usar apenas o bloco 0–999 (teste rápido):
   ```bash
   mv thermometer12_0_999X.p thermometer12_X.p
   mv thermometer12_0_999y.p thermometer12_y.p
   ```

4. Classificação por família (WiSARD):
   ```bash
   cd projeto_malwisard
   python src/wisard_adware.py
   python src/wisard_trojan.py
   python src/wisard_worm.py
   python src/wisard_backdoor.py   # requer dataset com amostras de Backdoor
   python src/wisard_virus.py      # requer dataset com amostras de Virus
   ```

5. Experimentos avançados com vírus (Decision Tree, mental images, etc.):
   - Certificar que `import gist` foi removido de `src/virus.py`.
   - Rodar:
   ```bash
   cd projeto_malwisard
   python src/virus.py
   ```

6. Classificação global em 6 classes agregadas:
   ```bash
   cd projeto_malwisard
   python src/wisard_6.py
   ```

Esse pipeline permite:
- Mostrar o funcionamento completo do sistema com um subset reduzido (0–999) para fins de demonstração rápida.
- Escalar para o dataset completo, gerando todos os blocos com `thermometer12.py` e utilizando o `join_dataset.py` para reproduzir o cenário do artigo original.


