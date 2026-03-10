# Version History

## v2.0 - 2026-03-10

### Phase 2: Document Processing & Embedding

**Embedding Model:** TF-IDF (5,000 features, unigrams + bigrams)

**Artifacts:**
- Train embeddings: 3,341 samples × 5,000 dimensions
- Val embeddings: 716 samples × 5,000 dimensions  
- Test embeddings: 717 samples × 5,000 dimensions
- Vectorizer: Fitted TF-IDF model
- Index maps: Feature name mappings

**Validation Metrics:**
- Intra-class similarity: 0.2124 (baseline acceptable)
- Inter-class similarity: 0.0144 ✓ (threshold: < 0.15)
- Outlier rate: 2.01% ✓ (threshold: < 10%)

**Total Size:** ~92 MB

**Checksums:** See `checksums.sha256`

**Scripts:**
- p2.1: Document preprocessing
- p2.2: Text embedding generation
- p2.3: Embedding validation
- p2.4: Storage and versioning

---
