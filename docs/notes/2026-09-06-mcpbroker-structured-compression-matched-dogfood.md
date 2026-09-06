# Matched Dogfood Evaluation: Structured MCP Tool Compression

**Date:** 2026-09-06  
**Issue:** #11825  
**Parent:** #11806  
**Spine:** 445add88430cba3b9f59e14d42b0fda25dcb9f1a  

---

## Executive Summary

Structured MCP tool response normalization (whitespace compaction of exact structuredContent mirrors) was evaluated under a matched 26-task workload comparing the active default arm against the noop control arm (`FAK_COMPRESSOR=noop` / `CompressionOff`).

Key conclusions:
1. **100% Semantic Fidelity:** Zero semantic degradation across all evaluated cases. All keys, values, and types match uncompressed outputs (`reflect.DeepEqual` on parsed JSON).
2. **Byte Conservation:** Byte conservation holds without leakage: `input_bytes = output_bytes + bytes_saved`.
3. **Payload-Free Provenance:** All emitted receipts carry exact codec (`json-min`), stage identity (`mcpbroker.compact_structured`), reason token, byte metrics, and timing attribution, with zero raw payload bytes in metadata.
4. **Latency Budget:** Average transformation latency is < 0.1ms per invocation, well within the 5ms budget.
5. **Caller Opt-Out:** Caller-selected identity preserves original bytes with zero savings and without mutating broker or global environment state.

---

## Evaluation Arms

- **Default Arm:** Automatic structured JSON compaction enabled by default.
- **Noop Control Arm:** Normalization disabled (`FAK_COMPRESSOR=noop` or caller opt-out).

---

## Receipt Conservation & Provenance Invariant

The replayable receipt validator enforces:
- `input_bytes == output_bytes + bytes_saved`
- No raw tool content or decoded strings in receipt metadata
- Hold verdict issued on any semantic error or conservation mismatch
