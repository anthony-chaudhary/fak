# Shared token-cache retention

The duplication scanner persists content-addressed token windows at
`<git-common-dir>/fak/token-cache`. Git's common directory makes one cache available
to the main checkout and every worktree without putting generated files in the source
tree. `FAK_TOKEN_CACHE=off` (also `0`, `false`, or `no`) disables reads, writes, and
automatic maintenance; the scanner then follows its unchanged uncached path.

## Retention defaults and overrides

| Setting | Default | Environment override | Maintenance-command override |
|---|---:|---|---|
| Immutable JSON bytes | 256 MiB | `FAK_TOKEN_CACHE_MAX_BYTES` | `--max-bytes` |
| Immutable JSON entries | 10,000 | `FAK_TOKEN_CACHE_MAX_ENTRIES` | `--max-entries` |
| Atomic-write temp grace | 24 hours | `FAK_TOKEN_CACHE_TEMP_GRACE` | `--temp-grace` |

Overrides must be positive. Invalid, zero, or negative environment values fall back to
the defaults; invalid command values are rejected. The temp grace uses Go duration
syntax such as `30m`, `24h`, or `168h`.

Retention removes whole `.json` entries, oldest modification time first, with filename
as the deterministic tie-break. This preserves the most recently written working set;
`Get` deliberately does not rewrite modification times, so the policy is FIFO-by-write,
not an unmeasured LRU signal. Malformed JSON remains a cache miss and participates in
the same byte/count eviction order.

Only `.entry-*.tmp` files strictly older than the grace are stale. A fresh temp is an
active-write candidate and is never removed. The maintenance lock lives at
`<git-common-dir>/fak/token-cache-maintenance.lock`, outside the entry directory. Lock
acquisition is nonblocking: one process converges the bounds, while contending writers
keep tokenizing and report `skipped_lock_busy`. On Windows, an entry open without delete
sharing can refuse removal; on Unix, an open descriptor can outlive unlink. Either way,
the reader sees a whole immutable JSON object or a miss, and failed removals are counted
as `skipped_locked_files` with a `partial` verdict.

## Lifecycle and receipts

`tokencache.Open` performs startup recovery. After a cache-backed
`clonescan.BuildTreeIndex` writes one or more misses, the consumer invokes one coalesced
maintenance pass for the completed batch. It does not hold the maintenance lock across
tokenization or ordinary `Get`/`Put` operations.

Run the same native seam explicitly when diagnosing or maintaining a clone:

```bash
fak dup cache-maintain --repo .
fak dup cache-maintain --repo . --json
fak dup cache-maintain --repo . --max-bytes 268435456 --max-entries 10000 --temp-grace 24h --json
```

The command accepts a repository root, resolves its Git common directory, and refuses a
cache path that escapes that directory through path traversal or a symlink. It does not
accept an arbitrary deletion directory. Human and JSON receipts report the configured
limits, before/after entry counts and bytes, removed entries and bytes, stale-temp
counts/bytes, skipped locked files, and the final verdict.

## Disable and rollback

For an immediate operational rollback, set `FAK_TOKEN_CACHE=off`; duplication checks
remain correct and recompute token windows in memory. Remove the variable to restore
the default bounded cache. To roll back only a temporary tuning change, unset the three
retention override variables and rerun `fak dup cache-maintain --repo . --json`. No
manual deletion inside `.git` is required.
