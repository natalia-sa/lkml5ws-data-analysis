#!/usr/bin/env python3

import argparse
import pathlib

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("path", type=pathlib.Path)
    parser.add_argument('--match', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--number", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    if not args.path.is_file():
        raise RuntimeError(f"File not found: {args.path}")

    if args.number <= 0:
        raise ValueError("--number must be greater than zero")

    df = pd.read_parquet(args.path, columns=["commit_hash", "message_id"])

    if args.match:
        df = df.dropna(subset=["message_id"])
    else:
        df = df.loc[df["message_id"].isna()]

    if df.empty:
        print("No commit <-> Message-ID mappings found.")
        return

    number = min(args.number, len(df))

    sample = df.sample(n=number, random_state=args.seed)

    print(f"Total mappings: {len(df)}")
    print(f"Sample size:    {number}")

    if args.seed is not None:
        print(f"Random seed:    {args.seed}")

    print()

    for _, row in sample.iterrows():
        print(f"{row['commit_hash']:<40}", end=" ")
        if pd.notna(row["message_id"]):
            print('<->', end=" ")
            print(f"{row['message_id']}")


if __name__ == "__main__":
    main()
