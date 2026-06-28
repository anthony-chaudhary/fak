# Virtual skill environment — versioned page table for skills

This package implements C4 of the SKILL-LOADER-QUERY epic (#1103): a **versioned page table** that enables skills to run in multiple versions side-by-side, with hot-swap (remap) and rollback (inverse remap) capabilities.

## Architecture

The virtual skill environment is a page-table-like abstraction over skill versions:

- **v1/v2/v3 side-by-side**: Multiple versions of a skill can be resident simultaneously in the procedural cache (`contextq.SkillContextRecord.Version` already re-keys on version, so each version is a distinct cache slot).
- **Hot-swap = remap**: Flipping the active version is a page-table entry change (`Pin`), not a reload. In-flight invocations keep their pinned frame (the `ctxmmu` CAS pin guarantees a pinned page survives eviction); new invocations resolve the new version.
- **Rollback = inverse remap**: Restoring a prior version is just another page-table entry flip (`Unpin` or `Swap` back).
- **Blast radius pre-flip**: Before any swap, `ctxresidency.Query` reports exactly what will be invalidated by the flip (evictable spans' token and dependent-entry counts).

## The `Table` type

`Table` is the versioned page table:

```go
type Table struct {
    versions map[string]string // skill name → active version
    resolver Resolver          // how to resolve a skill name to a version
    mmu      *ctxmmu.MMU        // for blast-radius reads
    kvctx    *kvmmu.Context     // for blast-radius reads
}
```

### Operations

| Operation | Effect | Blast radius? |
|-----------|--------|---------------|
| `Pin(skill, version)` | Bind skill to explicit version | Reported before flip |
| `Unpin(skill)` | Revert to resolver-based version | Reported before flip |
| `Swap(skill, from, to)` | Atomic remap from `from` to `to` | Reported before flip |
| `ActiveVersion(skill)` | Read current version | — |
| `List()` | Snapshot all pinned skills | — |

### Hot-swap flow

1. **Before swap**: Skill is pinned to v1. New invocations resolve v1.
2. **Hot-swap**: `Pin(skill, "v2")` remaps the page-table entry.
3. **After swap**: New invocations resolve v2. In-flight invocations keep their pinned v1 frame (CAS pin guarantees survival).
4. **Rollback**: `Swap(skill, "v2", "v1")` restores v1, or `Unpin(skill)` falls back to resolver.

## Acceptance criteria (witnessed)

All three acceptance criteria from #1107 are witnessed by tests:

1. **v1 and v2 resident simultaneously, no cross-talk** → `TestVersionedPageTable_SideBySideV1V2`
2. **Swap mid-invocation does not disturb pinned frame** → `TestHotSwap_InFlightSurvives`
3. **Rollback restores prior version, blast radius pre-flip** → `TestRollback_InverseRemap`, `TestBlastRadius_PreFlip`

## Dependencies

- **C1** (`internal/capindex`): Provides the skill resolver (`Index()` + `Fault()`) that `DefaultResolver` is a stub for today.
- **C3** (`ctxresidency`, `ctxmmu`): Provides blast-radius reads via `ctxresidency.Query`.
- **C2** (`contextq`): Already ships the procedural cache with version re-keying (`SkillContextRecord.Version`).

## Future work

- **Precise blast radius**: Today, `blastRadius()` reports a conservative upper bound (all evictable spans' costs). C6's witness surface should scan the `ViewCache` for procedural views whose `labels[version]` matches the old version, reporting only those spans' blast radii.
- **Resolver implementation**: `DefaultResolver` is a stub pending C1; it should read the version from a skill's SKILL.md frontmatter.
- **CLI surface**: Add `fak skill pin|unpin|swap|list` commands (C7).