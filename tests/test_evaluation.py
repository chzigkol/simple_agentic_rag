from agentic_rag.evaluation import parse_doc_ids, score_retrieval


def test_parse_doc_ids_normalizes_floaty_csv_ids() -> None:
    assert parse_doc_ids("1.0|2|003") == ["1", "2", "003"]


def test_score_retrieval_hit_at_rank_two() -> None:
    score = score_retrieval(["9", "2", "3"], ["2"])

    assert score.precision_at_k == 1 / 3
    assert score.recall_at_k == 1.0
    assert score.hit_at_k == 1.0
    assert score.mrr == 0.5
