# Governed dbt Semantics Before Raw SQL Architecture

**Date:** 2026-09-03  
**Status:** Architectural Decision & Research Specification  
**Related Masters:** #10646 (removable capability slots), #10652 (dormant data slot), #10653 (dbt semantics)  
**Implementation Leaf:** `internal/dataslot` (`dbt.go`, `dbt_test.go`)  

---

## 1. Problem Frame & Motivation

In modern analytics and data engineering repositories, raw SQL queries are an anti-pattern when structured dbt project definitions exist:
1. **Blind Schema Probes**: Autonomous agents executing raw SQL (`SELECT * FROM table LIMIT 1`) against production or warehouse databases incur compute costs, risk schema contention, and miss upstream column transformations.
2. **Loss of Lineage Context**: Raw SQL cannot answer *where* a metric originated, what models depend on a table, or what tests validate a column's invariants.
3. **Dormant Governance**: When dbt project artifacts (`dbt_project.yml`, `target/manifest.json`) are present, models, columns, descriptions, and dependency DAGs can be fully inspected locally without dialing database endpoints or executing raw SQL.

---

## 2. Core Architecture: dbt Semantics Adapter

The `dataslot` subsystem implements the governed dbt semantics adapter:
* **Detection**: `dataslot.Detect` recognizes `dbt_project.yml` and `dbt_project.yaml`, advertising the `FamilyDBT` capability slot in `dormant:ready` status with zero network dials.
* **Semantic Ingestion**: `dataslot.ReadDBTSemantics(manifestPath)` parses compiled manifest nodes in-memory, computing the SHA-256 artifact digest and indexing:
  * Model definitions, descriptions, and column schemas.
  * Direct upstream model dependencies (`depends_on`).
  * Reverse downstream model dependents (`LineageDown`).
* **Raw SQL Separation**: Raw SQL remains a distinct, policy-fenced capability. Queries to dbt semantic graphs execute with:
  * `RawSQLDormant: true`
  * `ZeroNetwork: true`
  * Deterministic artifact digest attribution.

---

## 3. Comparison: dbt MCP vs. Native dataslot Ingestion

| Dimension | dbt MCP / External Runner | fak Native `dataslot` Adapter |
|---|---|---|
| **Daemon Requirement** | Requires running Python dbt RPC / MCP server | Zero daemon; in-process pure Go JSON parser |
| **Connection Binding** | Often requires warehouse credentials in profiles.yml | Fully offline; reads static manifest artifacts |
| **Lineage Resolution** | External API call | In-memory DAG index (`Lineage(model)`) |
| **Trust / Evidence** | Self-reported tool response | SHA-256 digested `DBTSemanticsReceipt` |

---

## 4. Verification & Witness

The implementation is witnessed by `internal/dataslot/dbt_test.go`:
* `TestReadDBTSemantics`: Proves that a local fixture manifest resolves multi-model DAG relationships, column lists, and bidirectional lineage (`fct_orders` upstream `stg_customers` + `stg_orders`; `stg_customers` downstream `fct_orders`) while asserting `RawSQLDormant: true` and `ZeroNetwork: true`.
* `TestDetectDBTProject`: Proves that `Detect` automatically identifies `dbt_project.yml` as a ready capability slot.
