# Deterministic AMOProf-to-Confluence publication rerun — 2026-08-12

**Status:** witnessed runbook derived from Confluence page `587137170`, version 2.  
**Problem centrality:** Stewardship — preserves a repeatable evidence-publication path rather than changing the kernel.  
**Value frame:** For benchmark operators / the problem is that a successful GPU run can still lose its profiler evidence or become irreproducible during publication / today the bundle, analysis, page, and attachments cross different machines and tools / better because this runbook makes each boundary manifest-driven and independently witnessed / witness is an exact page-body pull plus an attachment-set comparison.

This note records the reusable process and the failures encountered while publishing the
2026-08-12 Live Code Benchmark + AMOProf bundle. It deliberately omits private control-channel
identifiers, addresses, credentials, and internal hostnames. Resolve sanctioned nodes and the
private transport through [`docs/private-comms-channel.md`](../private-comms-channel.md) and
[`docs/fleet-compute-nodes.md`](../fleet-compute-nodes.md).

## Proven reference result

| Item | Witnessed value |
|---|---|
| Confluence page | ID `587137170`, space `MPL`, version 2 |
| Source archive | `lcb-amoprof-final-20260812.tgz` |
| Archive bytes | `3,056,753` |
| Archive SHA-256 | `d2b72b0eafc0ce1e922091ac99e9a2551c9ea4debabca0b2a134a278c0567ebc` |
| Archive members | 74 files |
| Published attachments | 76 unique names: 74 members + original archive + attachment manifest |
| Attachment reconciliation | expected 76, actual 76, missing 0, extra 0 |
| Benchmark completion | scheduled 32, completed 32, failed 0 |
| Latency | mean 4.2426875 s; p50 4.393 s |
| Model | `google/gemma-4-31b-it` |
| fak revision | `793e38a87d71022070b807a1b0da809f4b86bc33` |

These are reference-run facts, not defaults for a future run.

## The deterministic state machine

Treat publication as five independently witnessed phases. Never infer that a later phase makes an
earlier one true.

1. **CAPTURED** — benchmark and profiler exit; immutable bundle exists on the compute node.
2. **TRANSFERRED** — publisher node has the same byte count and SHA-256; `gzip -t` succeeds.
3. **ANALYZED** — required outputs exist, collection validation passes, evidence boundaries are named.
4. **PUBLISHED** — page update and attachments return success.
5. **READ_BACK** — a fresh page pull and attachment API listing exactly match the declared contract.

A rerun is complete only at `READ_BACK`. Command output saying “Attached” or “Updated” is not the
final witness.

## Inputs to pin before running

Capture these in a run manifest before launching:

- benchmark date/run ID and UTC start;
- model ID and immutable weights revision;
- fak commit;
- prompt/generation corpus digest (`generations.json`);
- request count, concurrency, timeouts, and serving flags;
- AMOProf version/commit, collector set, cadence, and duration;
- compute topology and GPU model/count (publication-safe description only);
- expected output contract;
- target Confluence space, title, and existing page ID (if updating).

Change one input module at a time. A comparison is invalid when workload, serving, collection, and
analysis all drift together without separate manifests.

## Phase 1 — capture and seal on the compute node

Run the committed entrypoint and preserve its own logs:

```bash
set -euo pipefail
run_root="/path/to/run-${RUN_ID}"
mkdir -p "$run_root"
cd "$run_root"
bash run.sh >run.stdout.log 2>run.stderr.log
```

Validate before packaging:

```bash
test -s summary.json
test -s amoprof-run/collection_output_manifest.json
test -s amoprof-run/collection_validation.json
test -s amoprof-run/analysis/analysis_manifest.json
test -s amoprof-run/analysis/metrics_by_ai_operation.csv
```

Generate a sorted member checksum ledger, then archive from the parent directory so paths are
stable:

```bash
find . -type f ! -name SHA256SUMS -print0 \
  | LC_ALL=C sort -z \
  | xargs -0 sha256sum > SHA256SUMS
cd ..
tar --sort=name --mtime="@$SOURCE_DATE_EPOCH" \
  --owner=0 --group=0 --numeric-owner \
  -czf "${RUN_ID}.tgz" "${RUN_ID}"
stat -c '%s %n' "${RUN_ID}.tgz"
sha256sum "${RUN_ID}.tgz"
gzip -t "${RUN_ID}.tgz"
```

`--sort`, fixed mtime, and normalized ownership make archive bytes reproducible. If preserving real
mtimes is part of the evidence, omit normalization and call the result *integrity-verifiable*, not
byte-for-byte reproducible.

## Phase 2 — transfer with independent read-back

The compute node and publisher node are separate boundaries. First prove where the archive actually
exists; do not assume the requested path belongs to the publisher node.

Use the sanctioned private bridge or another approved binary-safe transport. On both ends record:

```bash
stat -c '%s %n' "$archive"
sha256sum "$archive"
gzip -t "$archive"
```

The two byte counts and hashes must match. Do not send a binary archive through chat text/base64:
message chunking, transcript limits, rate limits, and link rewriting make that path slow and fragile.
A temporary file server is acceptable only on the private lab network, for the minimum transfer
window, and must be stopped after the publisher confirms the hash.

## Phase 3 — inventory, validate, and analyze

Extract into a fresh directory and verify the archive ledger:

```bash
rm -rf "$extract_root"
mkdir -p "$extract_root"
tar -xzf "$archive" -C "$extract_root"
cd "$extract_root/$RUN_ID"
sha256sum -c SHA256SUMS
```

Gate the analysis on collection validity. At minimum prove:

- consolidated CSV and JSONL exist;
- required collectors have non-empty outputs;
- missing-file and empty-timeseries lists are understood;
- profiler start/end overlap the benchmark window;
- GPU count/model match the run manifest;
- operation attribution is either mapped or explicitly reported as unmapped.

For the reference run, 362 analysis samples were labeled `unmapped`. Therefore GPU, power, and I/O
measurements describe the sampled run window; they do **not** prove per-request or per-operation
causality. `power_efficiency_tok_per_wh = 0.0` was caused by absent token accounting, so tokens/Wh
was *not yet measured*.

Separate four claim classes on the page:

1. transport completion (`32 / 32 / 0`);
2. generated-code correctness (requires its own evaluator witness);
3. resource-envelope observations (AMOProf sampling window);
4. comparative efficiency/gain (requires a tuned baseline and net-true accounting).

Never promote class 1 into class 2, or class 3 into class 4.

## Phase 4 — deterministic attachment preparation

Confluence attachment names are a flat namespace. Preserve every archive member while avoiding
same-basename collisions by flattening paths deterministically:

```text
archive path:    amoprof-run/analysis/metrics_by_ai_operation.csv
attachment name: amoprof-run__analysis__metrics_by_ai_operation.csv
```

Reserve `__` as the path separator; escape a literal `__` in a source component (for example to
`--`) before flattening. Reject duplicate generated names before uploading.

Create `ATTACHMENT-MANIFEST.tsv` with:

```text
attachment_name<TAB>relative_path<TAB>bytes<TAB>sha256
```

Sort by relative path with `LC_ALL=C`. Include the original archive as a payload row. **Do not put
the manifest's own hash inside itself**: that is self-referential. Either (a) define the expected
attachment set as manifest rows plus the manifest filename, or (b) publish a detached
`ATTACHMENT-MANIFEST.sha256` and include both names in a small outer publication receipt.

Upload in sorted order and retain the command's complete output:

```bash
mapfile -d '' files < <(find "$flat_dir" -maxdepth 1 -type f -printf '%f\0' | LC_ALL=C sort -z)
confluence attach "$PAGE_ID" "${files[@]}"
```

Re-running `attach` should update same-name attachments, not mint timestamped duplicates. The final
API reconciliation catches either behavior.

## Phase 5 — page publication

The page should contain, in this order:

1. verdict and strongest supported result;
2. model, code revision, workload interval, and latency;
3. important GPU/power/energy findings;
4. interpretation and limitations;
5. archive hash and attachment-manifest contract;
6. deterministic rerun steps;
7. modular change protocol;
8. explicit evidence boundaries.

Always run the canonical dry-run first:

```bash
confluence push page.xhtml --dry-run
confluence push page.xhtml
```

### Scrubber-unavailable exception

The canonical behavior is fail closed when the private publication scrubber cannot load. Restore the
scrubber and rerun whenever possible. The reference publication used a user-authorized, one-process,
in-memory bypass only because the exact source context had been explicitly prepared for publication.
The exception had these invariants:

- dry-run succeeded first;
- the page itself recorded the date, authorization, reason, and scope;
- no publisher source or persistent configuration changed;
- the bypass lived for one process only;
- page and attachments were independently read back afterward.

Do not turn this into a global environment switch or a default CLI flag. A durable implementation,
if later needed, should require an explicit reason, page ID, content digest, operator authorization,
audit record, and expiry; it should still run all non-private output-safety transforms.

## Phase 6 — independent completion gate

Pull into a new directory, not the source directory:

```bash
mkdir -p verify && cd verify
confluence pull "$PAGE_ID"
```

Assert the pulled body contains the result, analysis, archive SHA-256, deterministic rerun section,
modular-change section, evidence limits, and any publication exception note.

Then query the attachment API with a page size above the expected count. Compare sets, not counts
alone:

```text
expected = attachment_name values from manifest + ATTACHMENT-MANIFEST.tsv
actual   = unique attachment titles returned by the API
require actual == expected
require missing == empty
require extra == empty
```

Also witness the original archive's attachment byte count and the IDs/sizes of the manifest,
`SHA256SUMS`, and primary analysis CSV. For stronger verification, download each attachment and
compare its SHA-256 with the manifest; API metadata alone proves presence and size, not content.

## Failure lessons from the reference run

| Failure | Cause | Deterministic recovery |
|---|---|---|
| Archive absent on publisher | Path belonged to compute node | Prove host/path with `stat`; transfer; compare hash on both ends |
| Chat/base64 transfer truncated or rewrote data | Text transport is not binary-safe; links were auto-wrapped | Use approved binary transport; `gzip -t` and hash before proceeding |
| Control read-back timed out | transient transcript/read rate limits | Commands write durable files; poll slowly; do not rerun mutations until effect is read back |
| XHTML command was mangled | shell/metacharacter and message-size boundaries | write XHTML from a file or numbered chunks; verify byte count and required markers before push |
| `confluence push` failed after dry-run | private scrubber dependency unavailable at runtime | restore scrubber; only with explicit authorization use the one-process audited exception above |
| Attachment output was too long | dozens of per-file lines exceeded transcript usefulness | rely on exit code, manifest, and attachment API set reconciliation |
| Manifest attempted to describe itself | self-size/hash changes when its row is appended | exclude self from payload rows or use a detached outer digest |
| API count looked correct | counts can hide one missing + one extra | compare exact title sets and witness key attachment sizes |

## Modular rerun matrix

| Module changed | Hold fixed | Required comparison witness |
|---|---|---|
| Workload | model, serving flags, profiler | corpus digest; request count/concurrency; completion and correctness |
| Model/serving | workload and profiler | model revision; serve flags; latency/throughput and correctness |
| Collection | workload and serving | collector manifest; cadence/window; collection validation |
| Analysis | immutable raw profiler files | analysis code revision; manifest; mapped/unmapped counts |
| Publication | immutable bundle and analysis | page body digest; exact attachment set; fresh pull/API read-back |

## End-of-run receipt

Archive a small machine-readable receipt containing:

```json
{
  "schema": "fak-amoprof-publication-receipt/1",
  "run_id": "...",
  "archive": {"name": "...", "bytes": 0, "sha256": "..."},
  "page": {"id": "...", "space": "...", "version": 0},
  "attachments": {"expected": 0, "actual": 0, "missing": [], "extra": []},
  "body_witnesses": ["result", "analysis", "reproduction", "limitations"],
  "publication_exception": null
}
```

The receipt is the deterministic handoff for the next rerun. It does not replace the source bundle,
page pull, attachment manifest, or API evidence; it points to all four.
