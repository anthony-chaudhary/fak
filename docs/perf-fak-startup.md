---
title: "Where fak's 172 ms startup actually goes"
description: "A measured attribution of fak.exe's per-spawn cost: not binary size, not package init, but an O(journal) rescan of the usage log on every invocation — and the fix that removes it."
---

# Profile: `fak.exe` per-spawn cost (#5626)

**Status:** attributed and the named cost removed (2026-08-06) · **Host:** Windows 11 Pro
26200, Go 1.26.5 windows/amd64, Defender real-time protection ON · **Method:** warm
(4 discarded priming calls), n=15 timed calls per row, medians, all rows in one session
on one host.

Issue [#5626](https://github.com/anthony-chaudhary/fak/issues/5626) measured `fak.exe
version` at 172 ms against a 26 ms `CreateProcess` floor and located the cost in
"startup, not dispatch", naming package `init()`, embedded assets, and eager config
loads as the likely suspects. **All three of those suspects are wrong.** This is what
the measurement says instead.

## 1. Binary size is not the cost

The hypothesis was that a 65.5 MB image is expensive to map and expensive for Defender
to scan per spawn. Control: two no-op Go binaries, identical except that one carries a
59.3 MiB embedded blob the linker cannot eliminate.

| binary | size | median |
| --- | --- | --- |
| no-op Go program | 2.11 MiB | **13.2 ms** |
| no-op Go program + 59.3 MiB embedded blob | 59.33 MiB | **15.4 ms** |

A 28x larger image costs 2 ms. On this host, with real-time protection on, image size is
**not** a meaningful per-spawn cost — Defender's scan result is cached against the
unchanged file. Trimming the binary would have bought nothing.

## 2. Package `init()` is not the cost

`GODEBUG=inittrace=1`, 4 runs, 326 packages:

```
run 1: pkgs=326  sum_clock=17.7ms  last_init_completes_at=23ms
run 2: pkgs=326  sum_clock=22.0ms  last_init_completes_at=28ms
run 3: pkgs=326  sum_clock=21.1ms  last_init_completes_at=28ms
run 4: pkgs=326  sum_clock=26.0ms  last_init_completes_at=30ms
```

Every package `init()` in the binary has finished within ~23–30 ms of runtime start.
(The per-package clock has a ~0.5 ms quantisation floor on Windows, so `sum_clock` is an
over-estimate across 326 packages, not an under-estimate.) There is no eager registry or
config load worth making lazy: the whole of init is a small fraction of 172 ms.

## 3. The cost is the usage journal, and it is O(total journal size)

One environment toggle isolates it — `FAK_USAGE_LOG=off` skips the usage-journal append
that `recordUsage` performs on every invocation:

| configuration | `fak version` median |
| --- | --- |
| usage log on (21.24 MB journal) | 184.3 ms |
| `FAK_USAGE_LOG=off` | **28.8 ms** |
| usage log on again | 180.0 ms |

That is ~84 % of the wall clock in one place. Redirecting the journal with
`FAK_USAGE_LOG_PATH` shows why — the cost is **linear in the size of the journal**, a
file that only ever grows:

| journal size | `fak version` median |
| --- | --- |
| 0 MB | 33.0 ms |
| 8.34 MB | 85.6 ms |
| 21.24 MB | 184.3 ms |

**Root cause.** `usagelog.Open` recovers the hash-chain head (the last row's `seq` and
`hash`) so an append continues the chain rather than forking it. `recoverHead` did that
by scanning the journal from **byte 0**, `json.Unmarshal`-ing every row, and keeping only
the last one. The journal on the reference host had reached 21.24 MB, so every single
`fak` invocation re-parsed the machine's entire recorded invocation history — millions of
rows of JSON — to learn two fields from the final line. The cost grows without bound, and
every hook that shells `fak` pays it once per turn.

This is why `--help` looked like evidence for "startup, not dispatch": both verbs pay the
same fixed journal tax, so both were slow. The tax was never startup at all — it is
post-dispatch work that every verb happens to share.

## 4. The fix

`recoverHead` now starts its scan at a bounded tail window (`recoverTailWindow`, 256 KiB)
instead of at byte 0, seeking to the first row boundary inside it. The chain head is by
definition the *last* row, so every row before the window could only be parsed and thrown
away. A journal smaller than the window, a window containing no row boundary, or a final
row larger than the window all fall back to the original whole-file scan, so the chain
head is never lost and never forked.

Skipping the older rows means a torn line *older* than the window no longer stops the
scan. That changes only the already-corrupt case, and in the safer direction: the head
becomes the true last row, so appends continue the live chain instead of forking off a
truncated prefix. Detecting corruption was never `recoverHead`'s job — `Verify` owns
that, and `Open` is deliberately robust so a damaged log cannot brick a CLI invocation.

## 5. Result

Same host, same session, warm, n=15, medians. The journal is the real 21.24 MB one
copied from the reference host for both columns.

| command | before | after | change |
| --- | --- | --- | --- |
| `fak version` | 198.8 ms | **31.8 ms** | **6.3x faster** |
| `fak --help` | 251.5 ms | **120.2 ms** | **2.1x faster** |

And the cost no longer grows with the journal:

| journal size | before | after |
| --- | --- | --- |
| 0 MB | 33.0 ms | 34.8 ms |
| 8.34 MB | 85.6 ms | 31.3 ms |
| 21.24 MB | 184.3 ms | 33.6 ms |

`fak version` now sits at 31.8 ms against a 13.2 ms Go-process floor and 45.4 ms for
`git --version` on the same host — it is no longer the most expensive image on the box.

**Binary size, before and after:** 69,791,232 → 69,794,304 bytes (66.56 MiB → 66.56 MiB,
+3,072 bytes, +0.004 %). Size was never the lever and the fix does not move it.

## 6. What this profile does *not* fix

- **`fak --help` still costs 120.2 ms**, ~88 ms more than `version`. That residual is
  genuine help-rendering work in the verb table, and it survives this fix. It is
  dispatch, not startup — the opposite of what #5626 assumed. Worth its own issue.
- **The decision journal (`guard-audit.jsonl`, 23.58 MB on this host) has the same
  head-recovery shape** in `internal/journal.recoverHead`. It was not measured here and
  is out of scope for a one-fix issue, but it is the obvious next place to look.
- **Nothing was made resident.** The #5626 alternative — serving hot hook queries from
  the lifecycle socket — remains open and is now much less urgent for `fak` itself.
