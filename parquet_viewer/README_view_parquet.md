# Visualizador rapido dos arquivos .parquet (LKML5Ws)

Este script (`parquet_viewer/view_parquet.py`) serve para **espiar** o conteudo
dos arquivos `.parquet` do dataset sem precisar abrir tudo. Voce passa o caminho
de um arquivo e quantas linhas quer ver; ele mostra **o total de linhas do
arquivo** e imprime so as primeiras N linhas que voce pediu.

> **Importante:** rode todos os comandos abaixo a partir da pasta `dataset`
> (a raiz, onde ficam `.venv/` e `LKML5Ws/`), e nao de dentro de
> `parquet_viewer/`.

## O que tem dentro da pasta

Depois de descompactar, a pasta `LKML5Ws/` tem varias subpastas no formato
`list=<nome-da-lista>` (ex.: `list=ath12k`, `list=netdev`, `list=rcu`...).
Cada subpasta representa **uma mailing list diferente** do kernel Linux e
contem um unico arquivo: `list_data.parquet`, com os emails daquela lista.
Ou seja, o que diferencia uma pasta da outra e simplesmente **de qual lista de
discussao** vieram os emails.

## Pre-requisitos (so na primeira vez)

E preciso um ambiente virtual (venv) com a biblioteca `pyarrow` instalada.
Rode os comandos abaixo dentro da pasta `dataset`:

```bash
python3 -m venv .venv
./.venv/bin/pip install pyarrow
```

> Isso cria a pasta `.venv/` e instala o `pyarrow` ali dentro, sem mexer no seu
> Python do sistema. Voce so faz isso **uma vez**.

## Como usar

Sempre rode o script com o Python do venv (`./.venv/bin/python`):

```bash
./.venv/bin/python parquet_viewer/view_parquet.py CAMINHO_DO_PARQUET [-n NUM_LINHAS] [-c COLUNAS] [--full]
```

Opcoes:

- `CAMINHO_DO_PARQUET` — caminho do arquivo `.parquet` (obrigatorio).
- `-n NUM_LINHAS` — quantas linhas mostrar (padrao: 10).
- `-c COLUNAS` — lista de colunas separadas por virgula (padrao: todas).
- `--full` — nao corta o texto das celulas longas (por padrao corta em 120 caracteres).

### Exemplos

Ver as 10 primeiras linhas (todas as colunas) e o total de linhas do arquivo:

```bash
./.venv/bin/python parquet_viewer/view_parquet.py "LKML5Ws/list=ath12k/list_data.parquet"
```

Ver as 20 primeiras linhas:

```bash
./.venv/bin/python parquet_viewer/view_parquet.py "LKML5Ws/list=ath12k/list_data.parquet" -n 20
```

Ver so algumas colunas (mais legivel):

```bash
./.venv/bin/python parquet_viewer/view_parquet.py "LKML5Ws/list=rcu/list_data.parquet" -n 5 -c from,subject,date
```

Ver o corpo completo do email sem cortar o texto:

```bash
./.venv/bin/python parquet_viewer/view_parquet.py "LKML5Ws/list=rcu/list_data.parquet" -n 1 -c subject,raw_body --full
```

> Dica: se voce nao lembrar os nomes das colunas, peca uma coluna qualquer
> errada (ex.: `-c xxx`) que o script lista todas as colunas disponiveis.

## Colunas principais

Os nomes usam `_` (underscore). As mais uteis no dia a dia:
`from`, `to`, `cc`, `subject`, `date`, `x_mailing_list`, `raw_body`,
`message_id`, `in_reply_to`. O esquema completo esta no `README.md` original
do dataset.
