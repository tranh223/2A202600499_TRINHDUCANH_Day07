from __future__ import annotations

from pathlib import Path

from src.chunking import ChunkingStrategyComparator


def infer_preserves_context(strategy_name: str, avg_length: float) -> str:
    if strategy_name == "fixed_size":
        return "Mostly" if avg_length >= 120 else "No"
    if strategy_name == "by_sentences":
        return "Yes" if avg_length >= 80 else "Mostly"
    if strategy_name == "recursive":
        return "Yes"
    return "N/A"


def run_experiment() -> None:
    comparator = ChunkingStrategyComparator()
    files_to_test = [
        "data/thanhtoan.md",
    ]

    print("| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |")
    print("|-----------|----------|-------------|------------|-------------------|")

    strategy_map = {
        "fixed_size": "FixedSizeChunker",
        "by_sentences": "SentenceChunker",
        "recursive": "RecursiveChunker",
    }

    for file_path in files_to_test:
        path = Path(file_path)
        if not path.exists():
            print(f"Không tìm thấy file {file_path}")
            continue

        text = path.read_text(encoding="utf-8")
        results = comparator.compare(text, chunk_size=200)
        doc_name = path.name

        for strategy in ("fixed_size", "by_sentences", "recursive"):
            stats = results[strategy]
            preserves_context = infer_preserves_context(strategy, stats["avg_length"])
            print(
                f"| {doc_name} | {strategy_map[strategy]} | "
                f"{stats['count']} | {stats['avg_length']:.2f} | {preserves_context} |"
            )


if __name__ == "__main__":
    run_experiment()