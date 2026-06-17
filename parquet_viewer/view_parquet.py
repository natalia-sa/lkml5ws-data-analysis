#!/usr/bin/env python3
"""Visualizador rapido de arquivos .parquet do dataset LKML5Ws.

Recebe o caminho de um arquivo .parquet e mostra:
  - o total de linhas do arquivo (sem carregar tudo na memoria);
  - as primeiras N linhas (N escolhido por voce).

Rode a partir da pasta `dataset` (raiz, onde ficam .venv/ e LKML5Ws/).

Uso:
    ./.venv/bin/python parquet_viewer/view_parquet.py CAMINHO [-n NUM_LINHAS] [-c col1,col2,...] [--full]

Exemplos:
    ./.venv/bin/python parquet_viewer/view_parquet.py "LKML5Ws/list=ath12k/list_data.parquet"
    ./.venv/bin/python parquet_viewer/view_parquet.py "LKML5Ws/list=ath12k/list_data.parquet" -n 20
    ./.venv/bin/python parquet_viewer/view_parquet.py "LKML5Ws/list=rcu/list_data.parquet" -n 5 -c from,subject,date
"""

import argparse
import sys

import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(
        description="Mostra o conteudo de um arquivo .parquet ate um numero de linhas.",
    )
    parser.add_argument("path", help="Caminho para o arquivo .parquet")
    parser.add_argument(
        "-n", "--rows", type=int, default=10,
        help="Quantidade de linhas para mostrar (padrao: 10)",
    )
    parser.add_argument(
        "-c", "--columns", default=None,
        help="Colunas a mostrar, separadas por virgula (padrao: todas)",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Nao corta o texto das celulas longas (por padrao corta em 120 caracteres)",
    )
    args = parser.parse_args()

    try:
        pf = pq.ParquetFile(args.path)
    except Exception as exc:  # arquivo inexistente, corrompido, etc.
        print(f"Erro ao abrir '{args.path}': {exc}", file=sys.stderr)
        sys.exit(1)

    total_rows = pf.metadata.num_rows
    all_columns = pf.schema_arrow.names

    # Resolve quais colunas mostrar.
    if args.columns:
        wanted = [c.strip() for c in args.columns.split(",") if c.strip()]
        invalid = [c for c in wanted if c not in all_columns]
        if invalid:
            print(f"Colunas inexistentes: {', '.join(invalid)}", file=sys.stderr)
            print(f"Colunas disponiveis: {', '.join(all_columns)}", file=sys.stderr)
            sys.exit(1)
        columns = wanted
    else:
        columns = all_columns

    n = max(args.rows, 0)

    print(f"Arquivo : {args.path}")
    print(f"Total de linhas : {total_rows}")
    print(f"Total de colunas: {len(all_columns)}")
    print(f"Mostrando ate {n} linha(s).")
    print(f"Colunas: {', '.join(columns)}")
    print("-" * 80)

    if n == 0 or total_rows == 0:
        return

    # Le apenas os primeiros lotes ate juntar n linhas (nao carrega o arquivo todo).
    collected = []
    for batch in pf.iter_batches(batch_size=min(n, 1000), columns=columns):
        rows = batch.to_pylist()
        collected.extend(rows)
        if len(collected) >= n:
            break
    collected = collected[:n]

    max_width = None if args.full else 120
    for i, row in enumerate(collected):
        print(f"===== Linha {i + 1} =====")
        for col in columns:
            value = row.get(col)
            text = str(value)
            if max_width is not None and len(text) > max_width:
                text = text[:max_width] + f"... [+{len(text) - max_width} chars]"
            print(f"  {col}: {text}")
        print()


if __name__ == "__main__":
    main()
