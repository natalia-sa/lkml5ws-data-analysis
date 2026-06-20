#!/usr/bin/env python3

import pandas as pd


FILES = [
    "amd-duplicated.csv",
    "iio-duplicated.csv",
]

N_THREADS_TO_SHOW = 5
BODY_CHARS = 1200
RANDOM_SEED = 0


def short(x, n=120):
    x = str(x or "").replace("\n", " ")
    return x[:n] + ("..." if len(x) > n else "")


def show_thread(df, thread_id):
    thread = df[df["_thread_id"] == thread_id].copy()

    if "date" in thread.columns:
        thread["date"] = pd.to_datetime(thread["date"], errors="coerce")
        thread = thread.sort_values("date")

    print()
    print("=" * 100)
    print(f"THREAD: {thread_id}")
    print(f"Emails: {len(thread)}")
    print("=" * 100)

    for i, (_, row) in enumerate(thread.iterrows(), start=1):
        print()
        print(f"EMAIL {i}/{len(thread)}")
        print("-" * 100)

        if "date" in row:
            print(f"Date:    {row.get('date')}")

        if "from" in row:
            print(f"From:    {row.get('from')}")

        print(f"Subject: {row.get('subject')}")

        if "raw_body" in row:
            print()
            print(short(row.get("raw_body"), BODY_CHARS))


def main():
    for path in FILES:
        print()
        print("#" * 100)
        print(f"FILE: {path}")
        print("#" * 100)

        df = pd.read_csv(path)

        if "_thread_id" not in df.columns:
            raise ValueError(f"{path} has no _thread_id column.")

        print(f"Rows/emails: {len(df)}")
        print(f"Threads:     {df['_thread_id'].nunique()}")

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            print(f"Date range:  {df['date'].min()} -> {df['date'].max()}")

        print()
        print("Largest threads:")
        print(df["_thread_id"].value_counts().head(10).to_string())

        first_emails = (
            df.sort_values("date") if "date" in df.columns else df
        ).groupby("_thread_id").head(1)

        print()
        print("Example first subjects:")
        print(first_emails["subject"].head(10).to_string(index=False))

        sample_threads = (
            first_emails["_thread_id"]
            .sample(
                n=min(N_THREADS_TO_SHOW, len(first_emails)),
                random_state=RANDOM_SEED,
            )
            .tolist()
        )

        print()
        print(f"Showing {len(sample_threads)} random threads for vibe check...")

        for thread_id in sample_threads:
            show_thread(df, thread_id)


if __name__ == "__main__":
    main()