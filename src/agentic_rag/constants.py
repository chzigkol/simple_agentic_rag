"""Project constants."""

from enum import StrEnum


class SourceType(StrEnum):
    """Supported retrieval routes."""

    RETRIEVE_QNA = "Retrieve_QnA"
    RETRIEVE_DEVICE = "Retrieve_Device"
    WEB_SEARCH = "Web_Search"


QNA_COLLECTION = "medical_qna"
DEVICE_COLLECTION = "medical_device_manual"

COLLECTION_BY_SOURCE: dict[SourceType, str] = {
    SourceType.RETRIEVE_QNA: QNA_COLLECTION,
    SourceType.RETRIEVE_DEVICE: DEVICE_COLLECTION,
}
