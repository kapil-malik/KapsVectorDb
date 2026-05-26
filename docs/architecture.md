## Architecture Evolution

This project intentionally evolves through multiple vector store implementations to explore the tradeoffs behind real-world vector databases.

### 1. `NaiveInMemVectorStore`

Baseline implementation.

- Stores vectors in a Python dictionary.
- Uses brute-force cosine similarity search.
- Computes similarity against every vector at query time.

Search complexity:

```text
O(N × D)
```

Where:
- `N` = number of vectors
- `D` = vector dimensions

This implementation prioritizes correctness and simplicity over performance.

---

### 2. `NormalizedInMemVectorStore`

Optimization over the naive implementation.

Key idea:
- Normalize vectors during insert.
- Normalize query vector during search.
- Use dot product similarity instead of full cosine similarity.

Instead of:

```text
cosine_similarity(a, b) = dot(a, b) / (||a|| × ||b||)
```

Search becomes:

```text
dot(normalized_query, normalized_vector)
```

This shifts work from the read path to the write path and significantly reduces search latency.

---

### 3. `MatrixBackedInMemVectorStore`

Introduces matrix-backed vector storage using NumPy.

Key idea:
- Store vectors in one contiguous matrix.
- Perform vectorized similarity computation.

Instead of:

```python
for vector in vectors:
    np.dot(query, vector)
```

Search becomes:

```python
scores = matrix @ query
```

This dramatically improves search performance using optimized native NumPy/BLAS operations.

Tradeoff:
- Insert performance becomes very poor because the matrix is repeatedly reallocated using `np.vstack(...)`.

---

### 4. `BufferedMatrixInMemVectorStore`

Improves the matrix-backed implementation using buffered inserts.

Key idea:
- Maintain:
    - a large immutable-ish matrix
    - a small mutable insert buffer
- Flush buffer into matrix periodically in batches.

Benefits:
- Retains fast vectorized matrix search.
- Avoids repeated matrix reallocations on every insert.
- Greatly improves insert throughput.

This design loosely resembles:
- memtables + SSTables
- buffered write paths in LSM-based systems

---

### 5. `FileBackedVectorStore`

Adds persistence layer to the buffered matrix store using files on disk.

Key idea:
- Maintain three files on disk:
    - records.jsonl: metadata and text for each record (one JSON per line)
    - vectors.npy: NumPy matrix of normalized vectors
    - tombstones.txt: list of deleted record IDs
- Keep an in-memory buffer for pending inserts.
- Periodically flush buffer to disk in batches.
- Support soft deletes via tombstoning.

### Store Comparison

| Store | Insert | Search | Persistence | Scalability |
|---|------------------------|---|---|-------------|
| `NaiveInMemVectorStore` | O(1) (append to dict) | O(N × D) (cosine similarity) | No | O(N) memory |
| `NormalizedInMemVectorStore` | O(D) (normalization) | O(N × D) (dot product) | No | O(N) memory |
| `MatrixBackedInMemVectorStore` | O(N × D) (matrix operations) | O(N × D) (vectorized operations) | No | O(N) memory | 
| `BufferedMatrixInMemVectorStore` | O(B × D) (buffered inserts) | O(N × D) (vectorized operations) | No | O(N) memory |
| `BufferedMatrixFileVectorStore` | O(B × D) (buffered inserts) | O(N × D) (vectorized operations) | Yes | Disk-backed |
