from pathlib import Path

import chromadb
import pandas as pd

from agentic_rag.constants import SourceType
from agentic_rag.ingestion import ensure_chroma_collections
from agentic_rag.retrievers import ChromaRetriever


async def test_ingestion_populates_chroma_and_retriever_reads_qna(
    tmp_path: Path,
) -> None:
    chroma_path = tmp_path / "chroma"
    client = chromadb.PersistentClient(path=str(chroma_path))
    qna_df = pd.DataFrame(
        [
            {
                "Question": "What are symptoms of dystonia?",
                "Answer": "Muscle contractions and repetitive movements.",
                "qtype": "symptoms",
            }
        ]
    )
    device_df = pd.DataFrame(
        [
            {
                "Device_Name": "Infusion Pump",
                "Model_Number": "IP-100",
                "Manufacturer": "Example Medical",
                "Patient_Population": "Adult",
                "Indications_for_Use": "Controlled medication delivery.",
                "Contraindications": "Do not use with incompatible tubing.",
                "Sterilization_Method": "Ethylene oxide",
            }
        ]
    )

    summary = ensure_chroma_collections(client, qna_df, device_df)
    retriever = ChromaRetriever(chroma_path=str(chroma_path), top_k=1)

    documents = await retriever.retrieve(
        SourceType.RETRIEVE_QNA,
        "What are symptoms of dystonia?",
    )

    assert summary.set_index("collection").loc["medical_qna", "count"] == 1
    assert summary.set_index("collection").loc["medical_device_manual", "count"] == 1
    assert documents[0].doc_id == "qna-0"
    assert "Muscle contractions" in documents[0].text
