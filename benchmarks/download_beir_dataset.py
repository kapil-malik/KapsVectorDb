"""
Download a BEIR dataset and convert it to the format expected by
benchmark_retrieval_quality.py (corpus.jsonl, queries.jsonl, qrels.tsv).

Usage:
    python benchmarks/download_beir_dataset.py \\
        --dataset scifact \\
        --output-dir data/retrieval_quality/beir/scifact

Recommended datasets for CPU-based sentence-transformer embedding (rough doc counts):
    scifact        ~5K docs    fast
    nfcorpus       ~4K docs    fast
    arguana        ~9K docs    fast
    trec-covid   ~171K docs    moderate
    fiqa          ~57K docs    moderate
    quora        ~523K docs    slow without GPU

For large datasets (nq, hotpotqa, msmarco, fever, dbpedia-entity) use
--max-corpus-docs and --max-queries to sample a manageable subset.
The corpus sample always includes all qrel-positive documents so retrieval
quality benchmarking remains valid.
"""

import argparse
import csv
import json
import random
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm


BEIR_BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"

BEIR_DATASETS = sorted([
    "arguana",
    "climate-fever",
    "dbpedia-entity",
    "fever",
    "fiqa",
    "hotpotqa",
    "msmarco",
    "nfcorpus",
    "nq",
    "quora",
    "scidocs",
    "scifact",
    "trec-covid",
    "webis-touche2020",
])


class _ProgressBar(tqdm):
    def update_to(self, count: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            self.total = total_size
        self.update(count * block_size - self.n)


def download_zip(dataset: str, dest: Path) -> Path:
    url = f"{BEIR_BASE_URL}/{dataset}.zip"
    zip_path = dest / f"{dataset}.zip"

    print(f"Downloading {url}")
    with _ProgressBar(unit="B", unit_scale=True, unit_divisor=1024, miniters=1) as bar:
        urllib.request.urlretrieve(url, zip_path, bar.update_to)

    return zip_path


def extract_zip(zip_path: Path, dest: Path) -> Path:
    print(f"Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)

    extracted = dest / zip_path.stem
    if extracted.is_dir():
        return extracted

    candidates = [d for d in dest.iterdir() if d.is_dir()]
    if len(candidates) == 1:
        return candidates[0]

    raise FileNotFoundError(
        f"Cannot locate extracted dataset directory in {dest}. "
        f"Found: {[d.name for d in candidates]}"
    )


def select_query_ids(
    qrels_path: Path,
    max_queries: int | None,
    sample_seed: int,
) -> list[str]:
    """Return sorted, deterministically sampled positive qrel query IDs."""
    all_ids: set[str] = set()
    with open(qrels_path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if int(row["score"]) > 0:
                all_ids.add(row["query-id"])

    sorted_ids = sorted(all_ids)
    if max_queries is None or max_queries >= len(sorted_ids):
        return sorted_ids
    return sorted(random.Random(sample_seed).sample(sorted_ids, max_queries))


def load_selected_qrels(
    qrels_path: Path,
    keep_query_ids: set[str],
) -> dict[str, dict[str, int]]:
    """Load positive qrels for selected query IDs, grouped by query."""
    qrels: dict[str, dict[str, int]] = {}
    with open(qrels_path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            query_id = row["query-id"]
            relevance = int(row["score"])
            if relevance <= 0 or query_id not in keep_query_ids:
                continue
            qrels.setdefault(query_id, {})[row["corpus-id"]] = relevance
    return qrels


def convert_corpus_safe(
    beir_dir: Path,
    output_path: Path,
    required_doc_ids: set[str],
    max_docs: int | None,
    sample_seed: int,
) -> tuple[int, set[str]]:
    """
    Write corpus.jsonl, always including all required_doc_ids (qrel-positive docs).
    Remaining slots up to max_docs are filled with deterministically sampled
    background docs. Returns (number of docs written, set of written doc IDs).
    """
    if max_docs is not None and max_docs < len(required_doc_ids):
        raise ValueError(
            f"--max-corpus-docs {max_docs} is smaller than the number of required "
            f"qrel-positive documents ({len(required_doc_ids)}). "
            f"Increase --max-corpus-docs or reduce --max-queries."
        )

    src = beir_dir / "corpus.jsonl"

    # Pass 1: collect all doc IDs to identify background candidates.
    # Only IDs are stored (not text) to avoid excessive memory use.
    all_doc_ids: list[str] = []
    with open(src, encoding="utf-8") as f:
        for line in tqdm(f, desc="Scanning corpus", unit=" docs"):
            line = line.strip()
            if not line:
                continue
            all_doc_ids.append(json.loads(line)["_id"])

    background_ids = [doc_id for doc_id in all_doc_ids if doc_id not in required_doc_ids]

    num_background = (
        len(background_ids)
        if max_docs is None
        else max_docs - len(required_doc_ids)
    )

    if num_background < len(background_ids):
        sampled_background = set(
            random.Random(sample_seed).sample(background_ids, num_background)
        )
    else:
        sampled_background = set(background_ids)

    selected_ids = required_doc_ids | sampled_background

    # Pass 2: stream corpus and write selected docs.
    written_ids: set[str] = set()
    with open(src, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="Writing corpus", unit=" docs"):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            doc_id = row["_id"]
            if doc_id not in selected_ids:
                continue
            title = row.get("title", "").strip()
            text = row.get("text", "").strip()
            combined = f"{title} {text}".strip() if title else text
            fout.write(json.dumps({"id": doc_id, "text": combined}) + "\n")
            written_ids.add(doc_id)

    missing_required = required_doc_ids - written_ids
    if missing_required:
        raise ValueError(
            f"{len(missing_required)} required qrel documents not found in corpus.jsonl: "
            f"{sorted(missing_required)[:5]}"
        )

    return len(written_ids), written_ids


def convert_queries(
    beir_dir: Path,
    output_path: Path,
    keep_ids: set[str],
) -> tuple[int, set[str]]:
    src = beir_dir / "queries.jsonl"
    written_ids: set[str] = set()

    with open(src, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row["_id"] not in keep_ids:
                continue
            fout.write(json.dumps({"id": row["_id"], "text": row["text"]}) + "\n")
            written_ids.add(row["_id"])

    return len(written_ids), written_ids


def write_qrels(
    selected_qrels: dict[str, dict[str, int]],
    output_path: Path,
    valid_query_ids: set[str],
    valid_doc_ids: set[str],
) -> tuple[int, int]:
    """Write qrels.tsv filtered to IDs that exist in the output corpus and queries."""
    written_query_ids: set[str] = set()
    pairs = 0

    with open(output_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(
            fout,
            fieldnames=["query_id", "doc_id", "relevance"],
            delimiter="\t",
        )
        writer.writeheader()

        for query_id, doc_rels in selected_qrels.items():
            if query_id not in valid_query_ids:
                continue
            for doc_id, relevance in doc_rels.items():
                if doc_id not in valid_doc_ids:
                    continue
                writer.writerow({
                    "query_id": query_id,
                    "doc_id": doc_id,
                    "relevance": relevance,
                })
                written_query_ids.add(query_id)
                pairs += 1

    return len(written_query_ids), pairs


def validate_output(output_dir: Path) -> None:
    corpus_ids: set[str] = set()
    with open(output_dir / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                corpus_ids.add(json.loads(line)["id"])

    query_ids: set[str] = set()
    with open(output_dir / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                query_ids.add(json.loads(line)["id"])

    qrel_query_ids: set[str] = set()
    qrel_doc_ids: set[str] = set()
    with open(output_dir / "qrels.tsv", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            qrel_query_ids.add(row["query_id"])
            qrel_doc_ids.add(row["doc_id"])

    print("\nValidation")
    print("----------")
    print(f"corpus docs      : {len(corpus_ids):,}")
    print(f"queries          : {len(query_ids):,}")
    print(f"qrel queries     : {len(qrel_query_ids):,}")
    print(f"qrel unique docs : {len(qrel_doc_ids):,}")

    errors = []
    if not corpus_ids:
        errors.append("corpus.jsonl is empty")
    if not query_ids:
        errors.append("queries.jsonl is empty")
    missing_queries = qrel_query_ids - query_ids
    if missing_queries:
        errors.append(
            f"{len(missing_queries)} qrel query IDs missing from queries.jsonl"
        )
    missing_docs = qrel_doc_ids - corpus_ids
    if missing_docs:
        errors.append(
            f"{len(missing_docs)} qrel doc IDs missing from corpus.jsonl: "
            f"{sorted(missing_docs)[:5]}"
        )

    if errors:
        raise ValueError("Validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    print("all checks passed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a BEIR dataset and convert it to benchmark_retrieval_quality.py format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=BEIR_DATASETS,
        metavar="DATASET",
        help=f"BEIR dataset name. Choices: {', '.join(BEIR_DATASETS)}",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        metavar="DIR",
        help="Directory to write corpus.jsonl, queries.jsonl, qrels.tsv",
    )
    parser.add_argument(
        "--split",
        choices=["test", "dev", "train"],
        default="test",
        help="qrels split to use (default: test)",
    )
    parser.add_argument(
        "--max-corpus-docs",
        type=int,
        default=None,
        metavar="N",
        help="Cap the corpus at N documents. Qrel-positive documents are always included.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        metavar="N",
        help="Sample N queries deterministically using --sample-seed.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed for deterministic query and corpus sampling (default: 42)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        zip_path = download_zip(args.dataset, tmp_dir)
        beir_dir = extract_zip(zip_path, tmp_dir)

        qrels_path = beir_dir / "qrels" / f"{args.split}.tsv"
        if not qrels_path.exists():
            available = sorted(p.stem for p in (beir_dir / "qrels").glob("*.tsv"))
            raise FileNotFoundError(
                f"No qrels file for split '{args.split}'. "
                f"Available splits: {available}"
            )

        print(f"\nConverting  : {args.dataset}  (split: {args.split})")
        print("-" * 50)

        # 1. Select query IDs deterministically.
        selected_query_ids = select_query_ids(qrels_path, args.max_queries, args.sample_seed)
        print(f"selected queries : {len(selected_query_ids):,}")

        # 2. Load positive qrels for selected queries.
        selected_qrels = load_selected_qrels(qrels_path, set(selected_query_ids))

        # 3. Collect required corpus doc IDs — must always appear in output corpus.
        required_corpus_ids = {
            doc_id for doc_rels in selected_qrels.values() for doc_id in doc_rels
        }
        print(f"required docs    : {len(required_corpus_ids):,}")

        # 4. Convert corpus, guaranteeing all required docs are included.
        corpus_count, written_corpus_ids = convert_corpus_safe(
            beir_dir,
            output_dir / "corpus.jsonl",
            required_doc_ids=required_corpus_ids,
            max_docs=args.max_corpus_docs,
            sample_seed=args.sample_seed,
        )
        print(f"corpus docs      : {corpus_count:,}")

        # 5. Convert queries.
        query_count, written_query_ids = convert_queries(
            beir_dir,
            output_dir / "queries.jsonl",
            keep_ids=set(selected_query_ids),
        )
        print(f"queries          : {query_count:,}")

        # 6. Safety check: no required qrel doc should be missing from the corpus.
        lost_pairs = [
            (qid, did)
            for qid, doc_rels in selected_qrels.items()
            for did in doc_rels
            if did not in written_corpus_ids
        ]
        if lost_pairs:
            raise ValueError(
                f"BUG: {len(lost_pairs)} required qrel doc IDs not in output corpus. "
                "This should not happen — please report."
            )

        # 7. Write qrels, filtered to IDs that were actually written.
        qrel_queries, qrel_pairs = write_qrels(
            selected_qrels=selected_qrels,
            output_path=output_dir / "qrels.tsv",
            valid_query_ids=written_query_ids,
            valid_doc_ids=written_corpus_ids,
        )
        print(f"qrel queries     : {qrel_queries:,}")
        print(f"qrel pairs       : {qrel_pairs:,}")

    # 8. Validate output files (outside tempdir — reads final output).
    validate_output(output_dir)

    print(f"\nOutput written to: {output_dir}")
    print("  corpus.jsonl")
    print("  queries.jsonl")
    print("  qrels.tsv")
    print("\nRun benchmark:")
    print(f"  python -m benchmarks.benchmark_retrieval_quality \\")
    print(f"    --dataset {output_dir} \\")
    print(f"    --store all \\")
    print(f"    --embedding-model sentence-transformer \\")
    print(f"    --top-k 10")


if __name__ == "__main__":
    main()