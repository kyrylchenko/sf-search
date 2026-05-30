# Embedding Query Quality Fix Plan

## Goal

Fix the local query UI ranking bug where text queries returned sky or unrelated
tiles above visibly relevant vehicle/road tiles.

## Evidence

- Existing image vectors were normalized and stored in the HNSW index.
- Directly comparing SigLIP text embeddings showed that `padding=True` reproduced
  the bad UI ranking, while `padding="max_length"` changed the top hits and put
  a road/car/tower tile first for the reported orange car query.
- Local torch reports no CUDA device but does report MPS availability, while the
  adapter selected only CUDA or CPU.

## Approach

- Add a regression test for the SigLIP text processor arguments.
- Add a regression test that automatic device selection prefers MPS before CPU.
- Update the adapter to use `padding="max_length"` and `truncation=True` for
  text queries.
- Update automatic device selection to prefer CUDA, then MPS, then CPU.
- Verify the existing local HNSW index through the same query service path. This
  does not require re-embedding images because only the text query vector
  changed.

## Verification

- `uv run pytest tests/embedding/test_embedding_model.py -q`
- `uv run pytest -q`
- Direct query probe against the current local index for
  `orange car far away on a highway near green tower`.
