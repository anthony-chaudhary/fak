# Notebooks — try `fak` in a hosted cloud notebook

Lowest-friction way to see the agent kernel work: no local Go toolchain, no clone on your
machine, a free GPU on demand. These are hosted demos over commands that already ship; the
quickstart is a modular case menu, and the in-kernel notebook is the deeper tier walk.

```mermaid
flowchart LR
    GEN["tools/gen_notebooks.py<br/>(generates both notebooks)"]
    QS["fak-quickstart.ipynb<br/>fresh modular cases"]
    IK["fak-inkernel.ipynb<br/>Lightning AI / RunPod"]
    A["Case A<br/>policy floor"]
    B["Case B<br/>HTTP adjudication"]
    C["Case C<br/>offline value measurement"]
    D["Case D<br/>real model gateway"]
    T2["Tier 2<br/>fused in-kernel decode + stable endpoint"]
    GEN --> QS
    GEN --> IK
    QS --> A
    QS --> B
    QS --> C
    QS --> D
    IK --> T2
```

*The generated notebooks and which proof/demo each one covers.*

| Notebook | Host | Covers | Status |
|---|---|---|---|
| [`fak-quickstart.ipynb`](fak-quickstart.ipynb) | **Google Colab** / Kaggle (free T4) | Policy proof, HTTP adjudication, offline value measurement, optional real-model gateway | runnable |
| [`fak-inkernel.ipynb`](fak-inkernel.ipynb) | Lightning AI / RunPod Jupyter (neocloud) | 2 — fused in-kernel decode + a stable endpoint | runnable |

Open the quickstart directly:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anthony-chaudhary/fak/blob/main/notebooks/fak-quickstart.ipynb)

## Running `fak-quickstart.ipynb`

1. Open it in [Google Colab](https://colab.research.google.com/) (or Kaggle), or run
   locally with `jupyter lab`.
2. **No key, no token.** `fak` is a public repo, so the *Get the binary* cell clones,
   refreshes to `FAK_BRANCH` (default `main`), and builds it anonymously. The cell prints
   the exact commit and `fak version` before the demos run.
3. Cases **A-C need no GPU**: policy proof, HTTP adjudication, and offline value
   measurement. For **Case D**, set **Runtime → Change runtime type → T4 GPU**, then
   ▶ **Run all**.

The notebook is **Run-all idempotent** (refreshes/re-builds the binary and re-pulls the
model on a fresh runtime) and degrades gracefully — with no GPU it runs the CPU cases and
tells you why Case D was skipped.

> **Knobs** (environment / Colab secrets): `FAK_REPO`, `FAK_BRANCH` (pin a release tag
> here), `FAK_REFRESH=0` (reuse an existing checkout without fetching), `FAK_MODEL`
> (default `qwen2.5:1.5b`), `FAK_WORK`, and `FAK_API_ADDR` (default `127.0.0.1:8765`).

## These notebooks are generated — don't hand-edit them

Both `.ipynb` files are **build artifacts** of [`../tools/gen_notebooks.py`](../tools/gen_notebooks.py).
The setup and *Get the binary* cells are identical across notebooks and live **once** in
that generator, so a change (a new flag, a pinned version, a clone tweak) updates every
notebook at the same time. The modular update path:

```bash
# edit a shared cell builder in tools/gen_notebooks.py, then:
python tools/gen_notebooks.py            # re-render notebooks/*.ipynb
python tools/gen_notebooks.py --check    # CI guard: fails on drift OR a stale repo reference
git commit -s -- tools/gen_notebooks.py notebooks/
```

`--check` is the anti-rot layer: it re-renders in memory and diffs against the committed
files (a "generated, do not edit" guard), **and** verifies every repo path/verb the cells
depend on still exists (`examples/…policy.json`, `testdata/…`, `scripts/fetch-model.sh`,
the `preflight` / `serve` / `agent` / `policy` / `attest` / `benchmarks` / `bench` /
`turntax` / `version` verbs) — so a refactor that removes one of those fails here
instead of failing a reader mid-notebook. Wire it into CI alongside the other lints.
