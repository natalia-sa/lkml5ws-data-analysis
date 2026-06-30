#!/usr/bin/env python3
"""
Classifies LKML emails regarding code duplication using GPT-5-mini.

Input:
    A parquet file containing LKML messages previously filtered by regex.

Output:
    A new parquet file with an additional column:

        duplication_classification

Possible values:

    CODE_DUPLICATION
    REDUNDANCY_REMOVAL
    NOT_CODE_DUPLICATION
    INSUFFICIENT_INFORMATION
    ERROR

The script preserves all original columns.
"""


import argparse
import json
import os
import time

import pandas as pd

from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI



# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL = "gpt-5-mini"
SAVE_EVERY = 50
REQUEST_DELAY = 0.2
CACHE_FILE = "llm_cache.json"


ALLOWED_LABELS = {
    "CODE_DUPLICATION",
    "REDUNDANCY_REMOVAL",
    "NOT_CODE_DUPLICATION",
    "INSUFFICIENT_INFORMATION"
}



# ---------------------------------------------------------
# Load API key
# ---------------------------------------------------------

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ---------------------------------------------------------
# LLM instructions
# ---------------------------------------------------------

SYSTEM_PROMPT = """
You are an expert in Software Engineering,
code clone detection and Linux Kernel development.

Your task is to classify whether an LKML email
is actually related to source code duplication.

The dataset was already filtered using regex.
Therefore many emails contain words such as:

duplicate
duplicated
duplication

but are false positives.

Analyze the technical meaning.

------------------------------------------------

CODE DUPLICATION

Code duplication occurs when the same or
substantially similar program logic exists
in multiple source code locations.

Clone taxonomy:

Type 1:
Exact duplicated code.

Type 2:
Duplicated code with renamed identifiers,
constants or types.

Type 3:
Copied code with modifications,
added or removed statements.

Type 4:
Different implementations providing
equivalent functionality.

------------------------------------------------

CODE DEDUPLICATION

Deduplication happens when duplicated
implementations are refactored into a
single shared implementation.

Examples:

- Extract duplicated logic into a helper.
- Replace multiple implementations by one common implementation.
- Share common initialization code.

------------------------------------------------

IMPORTANT

Removing redundancy is NOT code duplication.

Examples:

- Removing repeated function calls.
- Removing duplicated checks.
- Removing unnecessary conditions.
- Removing dead code.
- Cleanup.

These should be classified as:

REDUNDANCY_REMOVAL

------------------------------------------------

False positives:

These are NOT code duplication:

- duplicate index
- duplicate timestamp
- duplicate packet
- duplicate request
- duplicate message

unless the email explicitly discusses
duplicated source code.

------------------------------------------------

Return ONLY one label:

CODE_DUPLICATION
REDUNDANCY_REMOVAL
NOT_CODE_DUPLICATION
INSUFFICIENT_INFORMATION
"""



# ---------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------

def load_cache():

    if os.path.exists(CACHE_FILE):

        with open(CACHE_FILE) as f:
            return json.load(f)

    return {}



def save_cache(cache):

    with open(CACHE_FILE, "w") as f:
        json.dump(cache,f,indent = 2)



# ---------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------

def build_prompt(row):

    code = row.get("code", "")

    if pd.isna(code):
        code = ""


    return f"""

    Subject:

    {row.get("subject", "")}


    Email body:

    {row.get("raw_body", "")}


    Patch/code:

    {code}

    """



# ---------------------------------------------------------
# OpenAI classification
# ---------------------------------------------------------

def classify(row):

    response = client.responses.create(
        model=MODEL,
        input=[

            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },

            {
                "role": "user",
                "content": build_prompt(row)
            }

        ]

    )

    result = response.output_text.strip()

    if result not in ALLOWED_LABELS:
        return "INSUFFICIENT_INFORMATION"

    return result


# ---------------------------------------------------------
# Main processing
# ---------------------------------------------------------

def process(input_file, output_file):

    df = pd.read_parquet(input_file)
    cache = load_cache()
    classifications = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Classifying"):

        key = str(row.get("body_sha1", row.get("message_id")))

        if key in cache:
            label = cache[key]
        else:
            try:
                label = classify(row)
            except Exception as error:
                print(f"\nAPI error on row {idx}: {error}")
                label = "ERROR"
            cache[key] = label

        classifications.append(label)


        # Periodic checkpoint

        if idx % SAVE_EVERY == 0:
            temp = df.iloc[:len(classifications)].copy()
            temp["duplication_classification"] = classifications
            temp.to_parquet(output_file,index=False)
            save_cache(cache)

        time.sleep(REQUEST_DELAY)

    df["duplication_classification"] = classifications
    df.to_parquet(output_file,index=False)
    save_cache(cache)



    print(f"\nFinished.\nSaved: {output_file}")



# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input parquet file")
    parser.add_argument("output", help = "Output parquet file")

    args = parser.parse_args()

    process(args.input, args.output)



if __name__ == "__main__":
    main()
