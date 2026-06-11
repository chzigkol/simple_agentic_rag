from pathlib import Path

import pandas as pd

from agentic_rag.evaluation import load_examples, parse_doc_ids, score_retrieval


def test_parse_doc_ids_normalizes_floaty_csv_ids() -> None:
    assert parse_doc_ids("1.0|2|003") == ["1", "2", "003"]


def test_score_retrieval_hit_at_rank_two() -> None:
    score = score_retrieval(["9", "2", "3"], ["2"])

    assert score.precision_at_k == 1 / 3
    assert score.recall_at_k == 1.0
    assert score.hit_at_k == 1.0
    assert score.mrr == 0.5


def test_load_examples_resolves_qna_doc_ids_to_chroma_ids(tmp_path: Path) -> None:
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    pd.DataFrame(
        [
            {"Question": "other", "Answer": "other", "qtype": "information"},
            {"Question": "How common is condition X?", "Answer": "Rare.", "qtype": "frequency"},
        ]
    ).to_csv(datasets_dir / "medical_qna_dataset.csv", index=False)
    pd.DataFrame().to_csv(datasets_dir / "medical_device_manuals_dataset.csv", index=False)
    pd.DataFrame(
        [
            {
                "query": "How common is condition X?",
                "expected_source_type": "Retrieve_QnA",
                "expected_collection": "medical_qna",
                "expected_doc_ids": "0",
                "expected_answer": "Rare.",
                "category": "frequency",
            }
        ]
    ).to_csv(datasets_dir / "evaluation_dataset.csv", index=False)

    examples = load_examples(str(datasets_dir / "evaluation_dataset.csv"))

    assert examples[0].expected_doc_ids == ["qna-1"]


def test_load_examples_resolves_device_doc_ids_to_chroma_ids(tmp_path: Path) -> None:
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    pd.DataFrame().to_csv(datasets_dir / "medical_qna_dataset.csv", index=False)
    pd.DataFrame(
        [
            {
                "Device_Name": "Dialysis Machine",
                "Model_Number": "X-6538",
                "Patient_Population": "Adult",
                "Indications_for_Use": "Used for recovery.",
                "Contraindications": "None.",
            },
            {
                "Device_Name": "Dialysis Machine",
                "Model_Number": "X-0000",
                "Patient_Population": "Adult",
                "Indications_for_Use": "Used for recovery.",
                "Contraindications": "None.",
            },
        ]
    ).to_csv(datasets_dir / "medical_device_manuals_dataset.csv", index=False)
    pd.DataFrame(
        [
            {
                "query": "What is the intended use of the Dialysis Machine model X-6538?",
                "expected_source_type": "Retrieve_Device",
                "expected_collection": "medical_device_manual",
                "expected_doc_ids": "0",
                "expected_answer": "Used for recovery.",
                "category": "Indications_for_Use",
            }
        ]
    ).to_csv(datasets_dir / "evaluation_dataset.csv", index=False)

    examples = load_examples(str(datasets_dir / "evaluation_dataset.csv"))

    assert examples[0].expected_doc_ids == ["device-0"]
