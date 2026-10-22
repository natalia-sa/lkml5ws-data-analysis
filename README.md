# Code duplication in the Linux kernel mailing lists

This repository contains the pipeline that builds a dataset of mailing-list
threads about code duplication in the Linux kernel. It starts from the
[LKML5Ws](https://doi.org/10.5281/zenodo.17567225) dataset, groups the emails
into threads, selects and classifies the threads that deal with code
duplication, and links their patches to the commits that reached the mainline
kernel.

## Pipeline

| Step | Script | What it does |
|---|---|---|
| `fetch` | `fetch/fetch.py` | Downloads the LKML5Ws lists that appear in the kernel's `MAINTAINERS` file. |
| `build_threads` | `build_threads/build_threads.py` | Groups the emails of each list into threads, drops pull requests and bot messages, and flags the threads that match the regex pre-filter (`pre_filter/`). |
| `classify` | `classify/jev_classifier/classify_threads.py` | Classifies each flagged thread into one category with an LLM. |
| `commit` | `commit/commit_to_message_id.py` | Links the emails of the threads about duplication to mainline commits. |
| `publish` | `publish/publish_dataset.py` | Packs the output into tar archives and uploads them to an unpublished Zenodo draft. |

Every step reads and writes `data/output/list=<name>.parquet`, one file per
mailing list, adding its own columns. Every step also resumes from what is
already done.

## Setup

Requirements: Python 3.13, Git, and about 180 GB of free disk space for the
full dataset (156 GB of which are the downloaded lists).

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env                             
git submodule update --init repositories/linux    
```

`.env` holds the API key of the classifier (`TYPESAFE_API_KEY` for Jev) and
`ZENODO_TOKEN`, a Zenodo token with the `deposit:write` scope only. Paths,
the range of kernel releases and the Zenodo drafts are set in
`config/pipeline.yaml`.

## Running

```bash
.venv/bin/python run_pipeline.py                       
.venv/bin/python run_pipeline.py --lists b43-dev             
```

The `publish` step never publishes the dataset. It uploads the archives to a
draft, which is reviewed and published by hand on Zenodo. Create the draft
once with `publish/publish_dataset.py --create-draft` and set its id in
`config/pipeline.yaml`.

## Output

One row per thread.

| Column | Type | Description | Step |
|---|---|---|---|
| `thread_id` | string | Thread identifier. | `build_threads` |
| `list` | string | Mailing list. | `build_threads` |
| `n_messages` | int64 | Number of messages in the thread. | `build_threads` |
| `message_ids` | list\<string\> | Message-ID of each message, in chronological order. | `build_threads` |
| `date` | timestamp | Date of the first message. | `build_threads` |
| `subject` | string | Subject of the first message. | `build_threads` |
| `is_duplication_candidate` | bool | Whether the thread matches the regex pre-filter. | `build_threads` |
| `thread_content` | string | Text of the messages; null for non-candidates. | `build_threads` |
| `category` | string | Category given by the classifier; null for non-candidates and failed classifications. | `classify` |
| `category_confidence` | double | Probability of the chosen category, from 0 to 1. | `classify` |
| `commit_matches` | list\<struct\> | Commits linked to emails of the thread; null if the thread is not about duplication, empty if no email matched. | `commit` |

`category` is one of `clone_refactoring`, `preventive_reuse`,
`clone_and_reuse` (both of the previous two), `duplication_discussion` and
`not_duplication`. The categories are defined in
[`select_sample/LABELING_CRITERIA.md`](select_sample/LABELING_CRITERIA.md).

Each item of `commit_matches` links one email of the thread to one mainline
commit (tree at `af32da41b032`, 2026-10-09), not necessarily the patch about
duplication; `LINK` may also be a reference, such as the patch a fix
corrects. It has these fields:

| Field | Type | Description |
|---|---|---|
| `message_id` | string | Email of the thread linked to the commit. |
| `commit_hash` | string | Hash of the mainline commit. |
| `commit_subject` | string | Subject of the commit. |
| `commit_date` | timestamp | Author date of the commit, in UTC. |
| `commit_author` | string | Author name of the commit. |
| `status` | string | `LINK` (the commit's `Link:` trailer cites the email), `AUTHOR_DATE` (same author email and date), `AMBIGUOUS_LINK` or `AMBIGUOUS_AUTHOR_DATE` (more than one candidate email by that criterion). |
| `candidate_message_ids` | list\<string\> | Every candidate email, filled only for the `AMBIGUOUS_*` statuses. |

The remaining fields of each message, such as sender and recipients, can be
retrieved from LKML5Ws through `message_ids`.

## Labeled sample and classifier evaluation

`select_sample/` holds a sample of threads labeled by two reviewers, the
labeling criteria, and `sample_stats.ipynb`, which compares the three
classifiers in `classify/` (Jev, OpenJev and OpenAI) against the sample. Jev
is the classifier used by the pipeline.

## Tests

```bash
.venv/bin/python -m pytest classify commit/test_match_commits.py publish test_run_pipeline.py
.venv/bin/python -m pytest build_threads fetch pre_filter
```

## Repository layout

```
run_pipeline.py     runs the steps in order
config/             pipeline.yaml and cached allow-list data
fetch/              step 1: download of the lists
build_threads/      step 2: thread building
pre_filter/         regex pre-filter used by build_threads
classify/           step 3: classifiers, shared code and cache backup
commit/             step 4: link between emails and commits
publish/            step 5: upload to Zenodo
select_sample/      labeled sample, labeling criteria and evaluation notebook
repositories/linux  kernel repository (Git submodule)
data/               downloaded lists, output and archives (not versioned)
```

`patchwork/`, `notebooks/` and the compression scripts belong to an earlier
version of the analysis and are not part of the pipeline.
