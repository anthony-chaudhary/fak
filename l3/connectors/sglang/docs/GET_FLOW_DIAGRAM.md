# GET (Prefetch) Flow Diagram

```mermaid
flowchart TD
    A["LLM Request Arrives<br/>(SGLang Scheduler)"] --> B["prefetch() called<br/>Creates PrefetchOperation"]
    B --> C["Enqueue to prefetch_queue"]
    C --> D["Prefetch Thread picks up operation"]

    D --> E["_storage_hit_query()"]
    E --> F["Compute SHA256 hash<br/>per page of tokens"]
    F --> G["batch_exists(hashes)<br/>via _batch_exist()"]
    G --> H["conn.exists(key) × N<br/>16 thread-pool workers<br/>shared RDMA connection"]
    H --> I{{"storage_hit_count<br/>>= prefetch_threshold?"}}

    I -- No --> J["Revoke: put request_id<br/>on prefetch_revoke_queue"]
    J --> Z["Done — no prefetch"]

    I -- Yes --> K["Dispatch to prefetch<br/>executor thread pool"]
    K --> L["_page_transfer(operation)"]
    L --> M["For each batch of pages"]

    M --> N["batch_get_v1(hashes, host_indices)"]
    N --> N1["_apply_tag(hashes)<br/>optional prefix"]
    N1 --> N2["_batch_preprocess()<br/>expand to sub-keys + pointers"]
    N2 --> N3["_get_batch_zero_copy()"]

    N3 --> O["Build SGLs pointing into<br/>registered RDMA host buffer"]
    O --> O1{{"mget_rdma available?"}}

    O1 -- Yes --> P2["conn.mget_rdma(keys, sgls)<br/>1 control roundtrip<br/>+ batch RDMA Read (1 doorbell)<br/>+ 1 batch ack"]
    P2 --> Q["RDMA READ<br/>directly into host buffer<br/>(zero-copy, batch)"]

    O1 -- No --> P["conn.get(key, sgl, size) × N<br/>16 thread-pool workers<br/>(legacy fallback)"]
    P --> Q

    Q --> R["_batch_postprocess()"]
    R --> R1{{"MHA or MLA?"}}
    R1 -- "MHA (2 sub-keys/page)" --> R2["Both K and V must succeed<br/>→ per-page boolean"]
    R1 -- "MLA (1 sub-key/page)" --> R3["Single result per page<br/>→ per-page boolean"]
    R2 --> S["operation.increment<br/>(completed_tokens)"]
    R3 --> S

    S --> T{{"More batches?"}}
    T -- Yes --> M
    T -- No --> U["Host KV buffer filled"]

    U --> V["load() → CUDA async stream"]
    V --> W["Host buffer → GPU<br/>mem_pool_device"]
    W --> X["device_indices returned<br/>to attention computation"]
    X --> Y["LLM Inference proceeds"]

    Q -.->|"Timeout or error"| ERR["Increment _get_errors<br/>mark_terminate()"]
    ERR -.-> Z
```

## Key Points

- **No HTTP GET** — this is an internal prefetch pull, not a web request
- **Batch RDMA path (primary)**: `mget_rdma` sends all keys in one control roundtrip, posts all RDMA Reads with a single doorbell, and sends one batch ack. No ThreadPoolExecutor needed. Performance is bandwidth-limited and page-size-independent.
- **Legacy fallback**: If the server doesn't support `mget_rdma` (old version), falls back to per-key `get()` via ThreadPoolExecutor (16 workers)
- **Zero-copy RDMA**: NIC DMA-reads directly into the pre-registered host memory buffer — no CPU copies on either side
- **Existence check first** (`batch_exists`) avoids wasting RDMA bandwidth on missing keys
- **Threshold gate**: if too few pages hit in storage, the prefetch is revoked entirely
- **MHA vs MLA**: MHA stores separate K/V sub-keys (both must succeed), MLA uses a single fused sub-key per page
