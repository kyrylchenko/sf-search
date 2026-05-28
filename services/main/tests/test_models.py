from main_service.db.models.embedding import Embedding
from main_service.db.models.panorama import Panorama


def test_embedding_repr_is_closed() -> None:
    embedding = Embedding(id=123)

    assert repr(embedding) == "Embedding(id=123)"


def test_panorama_orig_id_column_accepts_real_google_id_length() -> None:
    orig_id_column = Panorama.__table__.c.orig_id

    assert orig_id_column.type.length >= 22
