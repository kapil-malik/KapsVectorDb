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


### 6. `MMapVectorStore`

Read-optimized vector store backed by a memory-mapped NumPy matrix.

Key idea:
- Load vectors using NumPy mmap instead of fully loading them into RAM.
- Allow the operating system to lazily page vector data from disk.

This store loads:
- metadata from records.jsonl
- vectors from vectors.npy using mmap_mode="r"
- tombstones from tombstones.txt

Benefits:
- Faster startup for persisted vector stores.
- Enables searching larger-than-memory vector datasets.

Tradeoff:
- Search latency may increase due to disk page faults and I/O.

### 7. `IVFVectorStore`

In-memory IVF (Inverted File Index) vector store.

Key idea:
- Partition vectors into clusters using KMeans.
- Search only a subset of clusters during query time.

Build process:
- Store normalized vectors.
- Build KMeans centroids.
- Assign each vector to its nearest centroid.
- Maintain inverted lists:
      centroid_id -> vector row indices

Search process:
- Find nearest `nprobe` centroids.
- Search only vectors belonging to those centroid lists.

Benefits:
- Significantly reduces vectors scanned per query.
- Improves search latency compared to brute-force search.

Tradeoff:
- Lower `nprobe` improves latency but reduces recall.

### 8. `FlatNSWVectorStore`

Flat Navigable Small World (NSW) graph-based ANN vector store.

Key idea:
- Maintain graph connections between neighboring vectors.
- Perform approximate search via graph traversal.

Insert process:
- Connect each new vector to nearby existing vectors.
- Maintain bounded neighbor lists per node.

Search process:
- Start from an entry point.
- Traverse increasingly similar neighbors using best-first search.

Benefits:
- Excellent recall vs latency tradeoff.
- Naturally supports incremental inserts.

Tradeoff:
- Higher memory usage due to graph edges.
- More complex insertion and graph maintenance logic.

This is currently a single-layer NSW graph, not full HNSW yet.

### 9. `HNSWVectorStore`

Hierarchical Navigable Small World (HNSW) graph-based ANN vector store.

Key idea:

* Organize vectors into multiple graph layers of increasing sparsity.
* Upper sparse layers enable fast long-distance navigation.
* Lower dense layers refine local nearest-neighbor search.

Insert process:

* Assign each vector a random maximum level.
* Navigate greedily from top layers toward the vector's region.
* Connect the vector to nearby neighbors on participating layers.

Search process:

* Start from the top-layer entry point.
* Perform greedy descent through upper layers.
* Run wider best-first search on the bottom layer.

Benefits:

* Excellent recall with low search latency at large scale.
* Faster navigation than single-layer NSW graphs.
* Incremental inserts without full index rebuilds.

Tradeoff:

* More complex graph maintenance and insertion logic.
* Additional memory overhead from multi-layer graph edges.
* Insert latency is higher than simpler ANN structures.


### Store Comparison

| Store | Insert | Search | Persistence | Scalability |
|---|------------------------|---|---|-------------|
| `NaiveInMemVectorStore` | O(1) (append to dict) | O(N × D) (cosine similarity) | No | O(N) memory |
| `NormalizedInMemVectorStore` | O(D) (normalization) | O(N × D) (dot product) | No | O(N) memory |
| `MatrixBackedInMemVectorStore` | O(N × D) (matrix operations) | O(N × D) (vectorized operations) | No | O(N) memory | 
| `BufferedMatrixInMemVectorStore` | O(B × D) (buffered inserts) | O(N × D) (vectorized operations) | No | O(N) memory |
| `FileBackedVectorStore` | O(B × D) (buffered inserts) | O(N × D) (vectorized operations) | Yes | Disk-backed |
| `MMapVectorStore` | O(B × D) (buffered inserts) | O(N × D) (vectorized mmap search) | Yes | Larger-than-memory datasets |
| `IVFVectorStore` | O(B × D) + build step | O((N / nlist) × nprobe × D) approximate search | No | ANN search with clustering |
| `FlatNSWVectorStore` | O(N × D) (graph neighbor discovery) | Approximate graph traversal | No | ANN search with graph index |
| `HNSWVectorStore` | O(log N) layer navigation + graph maintenance | Approximate hierarchical graph traversal | No | ANN search with hierarchical graph index |