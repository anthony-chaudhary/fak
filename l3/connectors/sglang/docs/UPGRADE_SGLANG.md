# Upgrading the bundled SGLang base

CAMA ships a pre-packaged SGLang tree (`sglang-with-cama-connector/`) that is
upstream SGLang plus a handful of CAMA patches. This is the runbook for rebasing
those patches onto a newer upstream release.

If you only want to know *what* CAMA changes in SGLang, read
[`patch_manifest.json`](../patch_manifest.json) — it lists every patched file and
why. The strategic background is in `docs/sglang-upgrade-plan.md` (repo root).

---

## The moving parts

| Thing | Role |
|---|---|
| `sglang-with-cama-connector/` | **Source of truth** for patched file *content*. Full SGLang tree + CAMA patches + the CAMA module. |
| `sglang-with-cama-connector/UPSTREAM.txt` | Pins the exact upstream commit the current patches sit on. Read by the tooling; rewritten by `upgrade-sglang.py`. |
| `cama-connector/patch_manifest.json` | **Source of truth** for *which* files are patched. |
| `cama-connector/patches/` | Generated mirror of the patched files, shipped in the standalone zip. `deploy.py` copies these onto a target tree. |
| `cama-connector/cama_module/` | The CAMA backend module, deployed to `…/storage/cama/`. Not a patch. |

Two rules keep this from drifting (both enforced in CI):

1. **`patches/` must equal the in-tree files** → `scripts/sync_patches.py --check`
2. **Every modified file must be in the manifest** → `scripts/find_cama_patches.py`

---

## Day-to-day: editing a patch

1. Edit the file in `sglang-with-cama-connector/python/sglang/…` (the source of truth).
2. `python scripts/sync_patches.py` — regenerates `patches/`.
3. Commit the in-tree change **and** the regenerated `patches/` together.

If you forget step 2, the `connector-patch-drift` CI job fails.

If you patch a *new* SGLang file, also add it to `patch_manifest.json` (with a
`why`). The `connector-patch-completeness` CI job fails until you do.

---

## Upgrading to a new upstream release

`upgrade-sglang.py` runs the rebase as two phases.

### Phase 1 — analyze (safe, no repo changes)

```bash
cd cama-connector
python upgrade-sglang.py v0.5.9
```

It clones the pinned base (from `UPSTREAM.txt`) and the target, then 3-way merges
every manifest file:

```
BASE   = upstream @ pinned base   (e.g. v0.5.7)
OURS   = in-tree patched file     (CAMA's version)
THEIRS = upstream @ target        (e.g. v0.5.9)
```

Output lands in `scratch/sglang-upgrade-<target>/`:

- `merged/<path>` — the merged files. **CLEAN** files are ready; **CONFLICT**
  files keep `<<<<<<< ours … ||||||| base … ======= … >>>>>>> theirs` markers.
- `MERGE_REPORT.md` — per-file status table.

`GONE` means upstream deleted or renamed the file — find where the logic moved
and re-apply by hand.

### Phase 2 — resolve, then apply (destructive, gated)

1. Edit each CONFLICT file under `merged/` until no markers remain.
2. Apply:

   ```bash
   python upgrade-sglang.py v0.5.9 --apply
   ```

   This refuses if any conflict markers are left (override with `--force` only if
   you know why). It builds a full new tree (full target checkout + your merged
   patches + the CAMA-added files preserved), **atomically swaps it in** (the old
   tree is kept as `sglang-with-cama-connector.old/`), rewrites `UPSTREAM.txt`,
   and regenerates `patches/`.

3. **Linux smoke test (hard prerequisite before committing).** A clean merge does
   *not* prove the v0.5.x-era patch logic still runs against the new upstream —
   internal SGLang APIs the patches call may have changed.

   ```bash
   python deploy.py /path/to/sglang-with-cama-connector --setup
   python -m sglang.launch_server --model-path <model> \
       --enable-hierarchical-cache --hicache-storage-backend cama
   # then run a basic inference request and confirm CAMA GET/SET works
   ```

4. Update `DIFF_STANDALONE_VS_SGLANG.md` (base ref/commit) and add a CHANGELOG
   entry. Delete `sglang-with-cama-connector.old/` once verified.

---

## Why the tooling forces LF

The in-tree tree is committed with LF. A fresh `git clone` on a machine with
`core.autocrlf=true` (Windows default) checks upstream out with CRLF. A naive
diff or merge then sees *every line* as changed — e.g. `environ.py` looked like
1181 changed lines when the real delta is 59, and an earlier hand-attempt at the
merge produced whole-file conflicts on the big files. All comparisons and merges
here normalize to LF first (`cama_patchlib.read_lf`, and `git -c
core.autocrlf=false` on clones), so the result is identical on Linux and Windows.

---

## Reference: the patched files

Run `python scripts/find_cama_patches.py` for the live list. As of the v0.5.7
base there are 8:

| File | Why |
|---|---|
| `srt/environ.py` | +CAMA env vars |
| `srt/server_args.py` | +CAMA CLI args |
| `srt/mem_cache/storage/backend_factory.py` | register the `cama` backend |
| `srt/managers/cache_controller.py` | zero-copy prefetch I/O + load-back backpressure |
| `srt/managers/schedule_policy.py` | wire `mem_quota` (load-back OOM guard) |
| `srt/managers/scheduler_metrics_mixin.py` | push scheduler metrics to the cache controller |
| `srt/mem_cache/hicache_storage.py` | `pp_rank`/`pp_size` + `prefix_keys` default fix |
| `srt/mem_cache/hiradix_cache.py` | HiRadix reconnect crash fix |
