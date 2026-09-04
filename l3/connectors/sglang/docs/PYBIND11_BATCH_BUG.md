# Pybind11 Batch Operations Bug

## The Problem

PrisKV's pybind11 wrappers for `mexists()`, `mset()`, `mget()` use **output parameters** (`std::vector<uint32_t>&`). Pybind11 automatically **copies** Python lists into temporary C++ vectors, so any modifications made on the C++ side are never reflected back to Python:

```
Python: status = [0, 0, 0, 0]  ──COPY──▶  C++: vector<uint32_t> status
                                            PrisKV writes: [0, 1, 0, 1]
Python: status = [0, 0, 0, 0]  (UNCHANGED)   C++ vector destroyed
```

**Impact:** `_batch_exist` always reports "all keys exist" → skips all writes → cache never populates.

## Status: Resolved

This bug is no longer relevant. CAMA's own Python client (`cama-client`) now implements batch operations directly at the wire protocol level, bypassing PrisKV's pybind11 wrappers entirely:

- **Existence checks:** `mexists()` → `OP_MTEST` (single roundtrip)
- **Writes:** `mset()` → `OP_MSET` (single roundtrip, with sub-batch chunking)
- **Deletes:** `mdel()` → `OP_MDEL` (single roundtrip)
- **Reads:** `mget_rdma()` → `OP_MGET_RDMA` + batch RDMA Read + `OP_BATCH_READ_ACK` (single control roundtrip + single RDMA doorbell for all keys, page-size-independent performance)

The `use_mput_mget` config flag (default `true`) controls whether the connector uses native batch ops. The `mget_rdma` capability is auto-detected via server handshake.

## Solutions (for fixing PrisKV's pybind11 bindings)

| Option | Approach | Complexity | Performance |
|--------|----------|-----------|-------------|
| **A (Recommended)** | Return `std::tuple<int, std::vector<uint32_t>>` as the function return value | Simple | Good — pybind11 handles return values natively |
| **B** | `PYBIND11_MAKE_OPAQUE(std::vector<uint32_t>)` — makes pybind11 pass the vector by reference instead of copying | Medium | Good — zero-copy on status vector |
| **C** | Accept `py::array_t<uint32_t>` (numpy array) — write directly into numpy-owned memory | Medium | Best — true zero-copy on status vector |

### Why Option A is recommended

- Minimal pybind11 code changes required
- No custom type handling or opaque declarations
- Pybind11 natively supports tuple return values — it "just works"
- Example signature: `std::tuple<int, std::vector<uint32_t>> mexists(const std::vector<std::string>& keys)`

## Upstream Status (as of March 2026)

**This is a known design limitation of pybind11, not a bug — it will NOT be fixed upstream.**

Pybind11's STL type casters (`pybind11/stl.h`) fundamentally copy Python containers into C++ temporaries. This is by design — there is no shared memory layout between a Python `list` and a `std::vector`, so a copy is unavoidable with the default casters. Multiple GitHub issues confirm the maintainers treat this as expected behavior:

- [Issue #4417 — "Pass by reference does copy"](https://github.com/pybind/pybind11/issues/4417) — Open, triaged, no fix planned.
- [Issue #1230 — "Cannot get std::vector from binded member function"](https://github.com/pybind/pybind11/issues/1230) — Confirmed as a design limitation.
- [Issue #2033 — "Problems passing std::vector by reference through virtual functions"](https://github.com/pybind/pybind11/issues/2033) — Even `PYBIND11_MAKE_OPAQUE` has edge-case failures with virtual function overrides.
- [Discussion #4340 — "modify std::vector as reference argument"](https://github.com/pybind/pybind11/discussions/4340) — Community workarounds only; no maintainer fix planned.

**Pybind11 v3.0.2** (latest, Feb 2026) does not address this. The v3.0.x changelog includes vectorcall performance improvements and Python 3.14 support but no changes to STL caster copy semantics.

### What about nanobind?

[Nanobind](https://github.com/wjakob/nanobind) (pybind11's successor by the same author) has the **same fundamental limitation** — STL type casters still copy. Nanobind's `bind_vector<>` can optionally use `rv_policy::reference_internal` for `__getitem__`, but this only affects element access, not function parameter passing. Migrating to nanobind would not fix this issue.

## Detailed Solution Analysis

### Option A — Return tuple (Recommended)

Change the C++ wrapper signature to return status via the function return value:

```cpp
// Before (broken)
int mexists(const std::vector<std::string>& keys, std::vector<uint32_t>& status);

// After (fixed)
std::tuple<int, std::vector<uint32_t>> mexists(const std::vector<std::string>& keys);
```

Pybind11 natively converts `std::tuple` → Python `tuple`, so the status vector is created on the C++ side and returned to Python as a new object. No copy-then-discard problem.

**Pros:** Simplest change, no special macros, no Python-side API changes beyond unpacking a tuple.
**Cons:** Allocates a new vector per call (negligible for status codes).

### Option B — `PYBIND11_MAKE_OPAQUE`

```cpp
PYBIND11_MAKE_OPAQUE(std::vector<uint32_t>)

// In module init:
py::bind_vector<std::vector<uint32_t>>(m, "UInt32Vec");
```

This disables the automatic list↔vector conversion and instead wraps the C++ vector as an opaque Python object. The Python side must use `UInt32Vec` instead of a plain list.

**Pros:** True pass-by-reference, zero-copy on the status vector.
**Cons:** Changes the Python API (callers must create `UInt32Vec` objects). Has known edge cases with virtual function overrides ([Issue #2033](https://github.com/pybind/pybind11/issues/2033)). Must be declared in every compilation unit before any usage.

### Option C — `py::array_t<uint32_t>` (NumPy)

```cpp
int mexists(const std::vector<std::string>& keys, py::array_t<uint32_t> status);
```

The C++ side writes directly into NumPy-owned memory. Python passes a pre-allocated `numpy.ndarray`.

**Pros:** Best performance — true zero-copy. NumPy arrays are familiar to Python users.
**Cons:** Adds a NumPy dependency. Requires the caller to pre-allocate with the correct dtype (`np.uint32`).

### Option D — Pass pointer instead of reference

A community workaround from [Issue #4417](https://github.com/pybind/pybind11/issues/4417): redirect reference parameters through a pointer-based wrapper.

```cpp
// Thin wrapper
int mexists_wrapper(const std::vector<std::string>& keys, std::vector<uint32_t>* status) {
    return mexists(keys, *status);
}
```

**Pros:** Minimal change to the existing binding.
**Cons:** Still requires `PYBIND11_MAKE_OPAQUE` for the vector to avoid copying; a pointer alone doesn't solve it.

## Historical re-enablement path (completed)

Native batch ops are now enabled by default. The pybind11 bug is bypassed entirely — CAMA's Python client encodes/decodes batch wire messages natively without going through PrisKV's pybind11 wrappers.

## References

- [pybind11 STL containers docs](https://pybind11.readthedocs.io/en/stable/advanced/cast/stl.html)
- [pybind11 changelog](https://pybind11.readthedocs.io/en/stable/changelog.html)
- [nanobind porting guide](https://nanobind.readthedocs.io/en/latest/porting.html)
- [pybind11 Issue #4417](https://github.com/pybind/pybind11/issues/4417)
- [pybind11 Issue #1230](https://github.com/pybind/pybind11/issues/1230)
- [pybind11 Issue #2033](https://github.com/pybind/pybind11/issues/2033)
- [pybind11 Discussion #4340](https://github.com/pybind/pybind11/discussions/4340)
