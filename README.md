# Análise de duplicação de código nas Linux Kernel Mailing Lists (LKML5Ws)

## Do que se trata

O ponto de partida é o dataset LKML5Ws, um conjunto **Apache Parquet particionado por mailing
list** (cada pasta `list=<nome>` guarda um `list_data.parquet` com os e-mails daquela lista).
O esquema completo do dataset está documentado em
[`parquet_viewer/README_dataset.md`](parquet_viewer/README_dataset.md).

A partir desses e-mails, o objetivo é chegar em um subconjunto confiável de threads sobre duplicação/deduplicação de código, classificá-las e verificar o desfecho de cada patch. Para isso o pipeline
combina três estratégias:

1. **Filtro por regex** (rápido, alto recall, muitos falsos positivos);
2. **Classificação por LLM** (interpreta o significado técnico da discussão);
3. **Consulta ao Patchwork** (descobre se o patch foi aceito no subsistema).

## Execução

### 1. Preparar o ambiente

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env  
```

Além das dependências do `requirements.txt` (`pandas`, `pyarrow`), a etapa de classificação por
LLM usa `openai`, `python-dotenv` e `tqdm`, e é a única que precisa da `OPENAI_API_KEY` em
`.env`.

### 2. Criar as pastas de dados

Estas pastas guardam dados de entrada/saída e ficam fora do controle de versão
(veja o `.gitignore`) — é preciso criá-las e populá-las localmente:

```bash
mkdir -p iio amd
```

### 3. Baixar e descompactar o dataset LKML5Ws

Baixe o `.tar.gz` do dataset (links e detalhes em
[Dataset e referências](#dataset-e-referências-lkml5ws), mais abaixo) e coloque-o na raiz
do repositório. Depois descompacte com o script incluso:

```bash
bash decompression_script.sh
```

Isso cria a pasta `LKML5Ws/`, particionada por lista (`LKML5Ws/list=<nome>/list_data.parquet`).

### 4. Preparar o parquet de entrada de cada subsistema

O pipeline espera um único `.parquet` por subsistema em `iio/list_data_iio.parquet` e
`amd/list_data_amd.parquet`. Copie as partições relevantes de `LKML5Ws/` para esses caminhos, por exemplo:

```bash
cp "LKML5Ws/list=<nome-da-lista-iio>/list_data.parquet" iio/list_data_iio.parquet
cp "LKML5Ws/list=<nome-da-lista-amd>/list_data.parquet" amd/list_data_amd.parquet
```

### 5. Rodar o filtro por regex

```bash
.venv/bin/python filter/filter_parquet.py iio/list_data_iio.parquet \
    --output filter/iio-duplicated.parquet
```

Detalhes do que esse script faz estão na seção [Como funciona](#como-funciona-o-pipeline)
abaixo.

### 6. Checar o resultado com o visualizador de parquet

```bash
.venv/bin/python parquet_viewer/view_parquet.py filter/iio-duplicated.parquet -n 10
```

Isso mostra o total de linhas do arquivo gerado e as 10 primeiras, para conferir rapidamente
se o filtro funcionou. Mais opções em [`parquet_viewer/README_view_parquet.md`](parquet_viewer/README_view_parquet.md).

## Como funciona o pipeline

O fluxo roda sobre os `.parquet` de cada subsistema e passa por estas etapas:

### 1. Filtragem por regex — `filter/filter_parquet.py`

Reconstrói as **threads** de discussão a partir dos cabeçalhos `In-Reply-To`/`References`
(usando *union-find*) e marca a thread inteira quando **qualquer** e-mail dela casa com uma
expressão regular de duplicação de código (`dedup`, `duplicate`, `copy-paste`, `clone`,
`redundant/identical/same code`, etc.), buscando tanto no assunto quanto no corpo/código.

Além de filtrar, o script:

- **liga as versões de um mesmo patch** (v2 → v1, v3 → v1, ...) na coluna
  `first_version_message_id`, agrupando submissões pelo autor + assunto-base;
- **sorteia ~10% das threads** para verificação manual (coluna `manual_verification`).

Saída: um `.parquet` só com as threads que mencionam duplicação de código. Comando de exemplo
na seção [Execução](#execução), passo 5.

### 2. Classificação por LLM — `classify/classify_duplication.py`

O regex tem muitos falsos positivos ("duplicate index", "duplicate packet"...). Esta etapa
envia **cada thread inteira** (uma chamada por thread) para o modelo **GPT-5-mini** da OpenAI,
que decide, a partir do significado técnico da conversa, um único rótulo:

| Rótulo | Significado |
|---|---|
| `CODE_DUPLICATION` | A thread discute duplicação de código-fonte (taxonomia de clones tipos 1–4). |
| `REDUNDANCY_REMOVAL` | Remoção de redundância/limpeza — **não** é duplicação de código. |
| `NOT_CODE_DUPLICATION` | Falso positivo do regex. |
| `INSUFFICIENT_INFORMATION` | Não dá para decidir. |

O rótulo é gravado na coluna `duplication_classification`. O script usa cache
(`classify/llm_cache.json`), checkpoints periódicos e chamadas paralelas, então pode ser
interrompido e retomado sem perder progresso.

```bash
.venv/bin/python classify/classify_duplication.py \
    filter/iio-duplicated.parquet classify/iio-regex-classified.parquet
```

> Requer uma chave `OPENAI_API_KEY` no arquivo `.env`.

### 3. Status de aceite no Patchwork — `patchwork/add_accepted_status.py`

Para cada *patchset*, o script encontra a linha do **último patch** da série e consulta o
**Patchwork** para descobrir se ele foi aceito no subsistema, preenchendo a coluna `accepted`
com o estado exato retornado pela API (`accepted`, `new`, `superseded`, `rejected`, ...).

Cada subsistema vive numa instância diferente do Patchwork:

- **iio** → `patchwork.kernel.org` (API REST moderna, `state` como string);
- **amd** → `patchwork.freedesktop.org` (fork antigo, API `1.0`, `state` numérico).

Detalhes, limitações e o significado exato de cada valor de `accepted` estão em
[`patchwork/README-add-accepted-status.md`](patchwork/README-add-accepted-status.md).

```bash
.venv/bin/python patchwork/add_accepted_status.py
```

### Scripts auxiliares

- **`parquet_viewer/view_parquet.py`** — espia rapidamente qualquer `.parquet` (total de
  linhas + primeiras N linhas). Ver [`parquet_viewer/README_view_parquet.md`](parquet_viewer/README_view_parquet.md).
- **`inspect/inspect_duplicated.py`** — imprime na íntegra as threads sorteadas para
  verificação manual.
- **`inspect/find_usp.py`** — localiza threads com pelo menos um remetente `@usp.br`.

## Notebook de análise

- **Rodar no Colab:** <https://colab.research.google.com/drive/1POtpQV_GusB20M8zl02bXNxkkAQAuIQZ?usp=sharing>
- **Versão no repositório:** [`notebooks/new_lkml5ws_data_analysis.ipynb`](notebooks/new_lkml5ws_data_analysis.ipynb)

## Estrutura do repositório

```
.
├── compression_script.sh / decompression_script.sh   # (des)compactação do dataset LKML5Ws
├── requirements.txt                                   # dependências (pandas, pyarrow, ...)
├── notebooks/
│   └── new_lkml5ws_data_analysis.ipynb                # notebook de análise (também no Colab)
├── parquet_viewer/                                    # visualizador + doc do esquema do dataset
├── iio/  amd/                                          # parquets de origem de cada subsistema
├── filter/                                            # 1. filtro por regex + threads + versões
│   └── filter_parquet.py
├── classify/                                          # 2. classificação por LLM (GPT-5-mini)
│   └── classify_duplication.py
├── patchwork/                                         # 3. status de aceite via Patchwork
│   ├── add_accepted_status.py
│   └── README-add-accepted-status.md
└── inspect/                                           # scripts auxiliares de inspeção
    ├── inspect_duplicated.py        # threads sorteadas para verificação manual
    └── find_usp.py                  # threads com remetentes da USP
```

> Os arquivos grandes de dados (`*.tar.gz`, `iio/`, `amd/`, parquets gerados, caches) ficam
> fora do controle de versão — veja o `.gitignore`.

## Dataset e referências (LKML5Ws)

### Sobre o dataset e o paper

- **MailingListsHeritage** — a ferramenta que criou o dataset:
  <https://gitlab.com/ccsl-usp/codev/MailingListsHeritage>
- **Pre-print do paper** (enviado para o ICSME 2026):
  <https://drive.google.com/file/d/1OOA3mq6BsoZuus7iOiKLZGW6WB7gdvhU/view?usp=sharing>

### Onde baixar o dataset

No dataset completo, **cada arquivo contém cerca de 20 listas**. A recomendação é escolher um
arquivo aleatório do dataset completo, ou usar o segundo link para baixar cada lista
individualmente.

- **Completo:**
  <https://zenodo.org/records/17567225?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjdmNWI2NWQ4LTZjZGUtNDczNS1hMzAyLWE3MWFhYTUyYjRjMiIsImRhdGEiOnt9LCJyYW5kb20iOiI0NzQ2YjAzZmEzZDNjMzFkODAzMWZjNzE3YWQ4NWFiZSJ9.LJgQGR2P2qjk04w4KpzCrLYjXKIMEbETa77YeG6S18YkzoDSktCDysVEZ9tqds4Qdh4SqNT29yDu_bKuvjYNoA&preview_file=README.md>
- **Separado por lista de e-mail:**
  <https://files.rcpassos.me/public/Academic/Datasets/LKML5Ws_uncompressed_lists/>

Para descompactar, ver a seção [Execução](#execução), passo 3. O esquema completo das colunas
está em [`parquet_viewer/README_dataset.md`](parquet_viewer/README_dataset.md).

### Editando a ferramenta MailingListsHeritage

Caso alguém queira **editar o código da ferramenta** usada para gerar o dataset, há um "pacote
inicial" com algumas listas já coletadas:

- **Pacote inicial de listas:**
  <https://files.rcpassos.me/public/Academic/MailingListsHeritage/>

Recomenda-se usar antes o **demo mode** (autocontido):
<https://gitlab.com/ccsl-usp/codev/MailingListsHeritage#demo-mode-self-contained>
