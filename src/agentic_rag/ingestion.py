"""Dataset ingestion helpers for local Chroma collections."""

from collections.abc import Iterable
from contextlib import suppress
from typing import Any, cast

import pandas as pd
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.api.types import PyEmbeddings
from sklearn.feature_extraction.text import HashingVectorizer

from agentic_rag.constants import DEVICE_COLLECTION, QNA_COLLECTION

EMBEDDING_DIMENSIONS = 512


def embed_texts(texts: Iterable[str]) -> PyEmbeddings:
    """Create deterministic local embeddings without external model downloads."""
    vectorizer = HashingVectorizer(
        n_features=EMBEDDING_DIMENSIONS,
        alternate_sign=False,
        norm="l2",
        ngram_range=(1, 2),
    )
    embeddings = vectorizer.transform(list(texts)).toarray().astype(float).tolist()
    return cast(PyEmbeddings, embeddings)


def ensure_chroma_collections(
    client: ClientAPI,
    qna_df: pd.DataFrame,
    device_df: pd.DataFrame,
    *,
    batch_size: int = 256,
    recreate: bool = False,
) -> pd.DataFrame:
    """Create and populate project Chroma collections when they are missing."""
    if recreate:
        for collection_name in [QNA_COLLECTION, DEVICE_COLLECTION]:
            with suppress(Exception):
                client.delete_collection(collection_name)

    qna_collection = _get_collection_with_expected_dimension(client, QNA_COLLECTION)
    device_collection = _get_collection_with_expected_dimension(
        client, DEVICE_COLLECTION
    )

    _populate_if_empty(
        qna_collection,
        _qna_records(qna_df),
        batch_size=batch_size,
    )
    _populate_if_empty(
        device_collection,
        _device_records(device_df),
        batch_size=batch_size,
    )

    collections = {
        collection.name: collection for collection in client.list_collections()
    }
    return pd.DataFrame(
        [
            {
                "collection": name,
                "exists": name in collections,
                "count": collections[name].count() if name in collections else 0,
            }
            for name in [QNA_COLLECTION, DEVICE_COLLECTION]
        ]
    )


def _get_collection_with_expected_dimension(
    client: ClientAPI,
    collection_name: str,
) -> Collection:
    collection = client.get_or_create_collection(
        collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    try:
        stored_dimension = _stored_embedding_dimension(collection)
    except Exception:
        stored_dimension = -1

    if stored_dimension is not None and stored_dimension != EMBEDDING_DIMENSIONS:
        client.delete_collection(collection_name)
        collection = client.get_or_create_collection(
            collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    return collection


def _stored_embedding_dimension(collection: Collection) -> int | None:
    if collection.count() == 0:
        return None

    result = collection.get(limit=1, include=["embeddings"])
    embeddings = result.get("embeddings")
    if embeddings is None or len(embeddings) == 0:
        return None

    first_embedding = embeddings[0]
    return len(first_embedding)


def _populate_if_empty(
    collection: Collection,
    records: list[dict[str, Any]],
    *,
    batch_size: int,
) -> None:
    if collection.count() > 0:
        return

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        documents = [str(record["document"]) for record in batch]
        collection.add(
            ids=[str(record["id"]) for record in batch],
            documents=documents,
            metadatas=[dict(record["metadata"]) for record in batch],
            embeddings=embed_texts(documents),
        )


def _qna_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    for row_index, row in df.reset_index(drop=True).iterrows():
        question = _clean_text(row["Question"])
        answer = _clean_text(row["Answer"])
        records.append(
            {
                "id": f"qna-{row_index}",
                "document": f"Question: {question}\nAnswer: {answer}",
                "metadata": _metadata(
                    {
                        "dataset": "medical_qna_dataset",
                        "row_index": row_index,
                        "qtype": row.get("qtype"),
                    }
                ),
            }
        )
    return records


def _device_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    document_columns = [
        "Device_Name",
        "Model_Number",
        "Manufacturer",
        "Patient_Population",
        "Indications_for_Use",
        "Contraindications",
        "Sterilization_Method",
    ]
    metadata_columns = [
        "Device_Name",
        "Model_Number",
        "Manufacturer",
        "Device_Class",
        "Publication_Date",
    ]

    for row_index, row in df.reset_index(drop=True).iterrows():
        parts = [
            f"{column}: {_clean_text(row[column])}"
            for column in document_columns
            if column in row and not pd.isna(row[column])
        ]
        records.append(
            {
                "id": f"device-{row_index}",
                "document": "\n".join(parts),
                "metadata": _metadata(
                    {
                        "dataset": "medical_device_manuals_dataset",
                        "row_index": row_index,
                        **{
                            column: row[column]
                            for column in metadata_columns
                            if column in row
                        },
                    }
                ),
            }
        )
    return records


def _metadata(values: dict[str, Any]) -> dict[str, str | int | float | bool]:
    metadata: dict[str, str | int | float | bool] = {}
    for key, value in values.items():
        if pd.isna(value):
            continue
        if isinstance(value, str | int | float | bool):
            metadata[key] = value
        else:
            metadata[key] = str(value)
    return metadata


def _clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())
