from typing import Any

from vectordb.models import VectorRecord


def metadata_matches(
        record: VectorRecord,
        filters: dict[str, Any] | None,
) -> bool:
    if not filters:
        return True

    for key, expected_value in filters.items():
        actual_value = record.metadata.get(key)

        if actual_value != expected_value:
            return False

    return True