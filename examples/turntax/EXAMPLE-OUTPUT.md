# Captured `fak turntax` output

Verbatim runs of the commands in [`README.md`](README.md), captured with the prebuilt
`fak` v0.34.0 binary (`go1.26.3 windows/amd64`) from the repo root. The fak side is **live
kernel events** — `turns_saved` per lever equals the kernel's own counters
(`Counters.Transforms`, `VDSOHits`), not a model. Re-running reproduces these numbers
exactly (the replay is deterministic); only the `1-shot serve p50` nanosecond figure is
wall-clock and will vary.

## 1. The real workload — all four classes fire

```console
$ fak turntax --trace examples/turntax/sample-trace.json
== fak turntax: turntax-walkthrough-sample  (14 calls, hash e3fe571cb0b5ef33) ==
consistency guard (counters==classification, not an independent oracle): ok

-- class breakdown (live kernel verdicts) --
  grammar repair (TRANSFORM)  : 2   -> saved 2 baseline reparse turns
  vdso tier-1 pure            : 2
  vdso tier-2 dedup (cache)   : 3
  vdso tier-3 static          : 2
  vdso total                  : 7   -> saved 7 baseline round-trip turns
  quarantine (poison held)    : 1   [safety floor]
  deny (capability floor)     : 1   [safety floor]
  pass (allow+engine, control): 3   -> 0 saved (both arms pay it)

-- NET turn-tax (happy path, deterministic) --
  turns saved                 : 9  (forced 5 = grammar+dedup; elision 4 = pure+static)
  tokens saved                : 11880  (@ 1200+120 tok/turn)
  dollars saved               : $0.04860
  latency saved               : 13.50 s  (@ 1500 ms/turn; 1-shot serve p50 = 3459 ns)

-- vDSO ablation (REAL ON/OFF path swap) --
  turns saved  vdso ON  : 9
  turns saved  vdso OFF : 2
  vdso lever contribution: 7 turns  (== VDSOHits 7)

-- safety floor (deterministic moat, NOT a turn count) --
  injections admitted   baseline=1  fak=0
  destructive executed  baseline=1  fak=0

-- ablation levers --
  grammar-repair [turn-tax] turns=2    TRANSFORM in-syscall (alias->canonical)
  vdso           [turn-tax] turns=7    3-tier local serve (pure / content-cache / static)
  quarantine     [safety-floor] turns=0    context-MMU result admission (poison paged out)
  deny           [safety-floor] turns=0    capability-floor adjudication (deny-as-value)
  NET (1-shot)   [turn-tax] turns=9    all levers — turns the baseline pays and fak does not

-- cost sensitivity (turns fixed by the kernel; per-turn price varies) --
  local-fast (400ms, 600tok)     tokens=6480     $0.03240  3.60s
  hosted-flash (1.5s, 1200tok)   tokens=11880    $0.04860  13.50s
  frontier (4s, 4000tok)         tokens=37080    $0.12420  36.00s

report written              : turntax-report.json
```

Read it as: **NET 9 turns saved**, decomposed into **forced 5** (the baseline is forced to
re-emit: grammar 2 + dedup 3) and **elision 4** (the kernel serves locally: pure 2 +
static 2). The vDSO lever is a *real* on/off path swap — `9 − 2 = 7`, exactly the kernel's
`VDSOHits`. The safety floor (1 quarantine, 1 deny) saves **0 turns** and is reported on its
own axis: it kept 1 injection and 1 destructive op out, neither folded into the dollars.

## 2. The anti-inflation control — a clean happy path saves exactly 0

```console
$ fak turntax --trace examples/turntax/sample-trace-happy.json
== fak turntax: turntax-walkthrough-happy  (3 calls, hash fa9aa7edfb366a75) ==
consistency guard (counters==classification, not an independent oracle): ok

-- class breakdown (live kernel verdicts) --
  grammar repair (TRANSFORM)  : 0   -> saved 0 baseline reparse turns
  vdso tier-1 pure            : 0
  vdso tier-2 dedup (cache)   : 0
  vdso tier-3 static          : 0
  vdso total                  : 0   -> saved 0 baseline round-trip turns
  quarantine (poison held)    : 0   [safety floor]
  deny (capability floor)     : 0   [safety floor]
  pass (allow+engine, control): 3   -> 0 saved (both arms pay it)

-- NET turn-tax (happy path, deterministic) --
  turns saved                 : 0  (forced 0 = grammar+dedup; elision 0 = pure+static)
  tokens saved                : 0  (@ 1200+120 tok/turn)
  dollars saved               : $0.00000
  latency saved               : 0.00 s  (@ 1500 ms/turn; 1-shot serve p50 = 4823 ns)

-- safety floor (deterministic moat, NOT a turn count) --
  injections admitted   baseline=0  fak=0
  destructive executed  baseline=0  fak=0
```

Three clean calls — no alias, no duplicate, no poison — and the kernel saves **nothing**.
This is the proof the headline reflects real avoided errors, not a fixed per-call discount.
It is what `TestRun_HappyPathSavesNothing` pins green.

## 3. Price your own hit-rate — the break-even sweep

```console
$ fak turntax --trace examples/turntax/sample-trace.json --breakeven
== fak turntax break-even: turntax-stochastic-base  (base 14 calls, 200 trials/point, hash db039fd8d1e103dd) ==
real-world addressable hit-rate (TURN-TAX §3.1, tau2-airline): 0.7%

     h   mean_turns   p50   p90   tok/sess     $/sess    lat(s)    self-host_sess
 0.000       0.0000     0     0          0    0.00000      0.00             never
 0.007       0.3250     0     1        429    0.00176      0.49        1595441596  <- real-world rate
 0.020       1.0600     1     2       1399    0.00572      1.59         489168414
 0.050       2.7300     3     5       3604    0.01474      4.09         189933524
 0.100       5.5750     5     8       7359    0.03011      8.36          93007807
 0.200      10.6100    11    14      14005    0.05729     15.91          48870737
 0.300      16.3700    17    20      21608    0.08840     24.55          31674925
 0.500      27.1800    28    32      35878    0.14677     40.77          19077209

The turn-saving is small at the real ~0.7% rate (the safety floor is the reason to run the kernel there);
it only becomes large in error/dup-rich regimes. The airline demo slice (9/14) is the far high end of this curve.
```

The demo slice (9 saved / 14 calls) is deliberately error-rich — the far right of this
curve. At a representative production error rate (`h ≈ 0.7%`) the per-session turn-saving is
fractional, which is *why* the report keeps the safety-floor axis separate: at low error
rates the floor is the reason to run the kernel, and the turn-tax is a modest cost win on
top. Price your own trace, not the demo.

---

Reproduce: run the three commands above from the repo root with the prebuilt binary
(`C:\Users\USER\bin\fak.exe` on Windows) or `go build -o fak ./cmd/fak` from a clean
checkout. The Go witnesses are `TestRun_HappyPathSavesNothing` and
`TestRun_VDSOAblationIsARealPathSwap` in `internal/turnbench/turnbench_test.go`.
