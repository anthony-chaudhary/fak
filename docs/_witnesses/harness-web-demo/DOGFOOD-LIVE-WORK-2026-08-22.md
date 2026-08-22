---
title: "Harness operator home real-work dogfood — 2026-08-22"
description: "Live repository-data readout for issue #8247, including the failing-before render, connected control-plane render, and filed defects."
---

# Harness operator home real-work dogfood — 2026-08-22

Issue: #8247. Captured implementation: `internal/harnessweb@r2+g4bcf91268a` from
committed source snapshot `56c32a0364e49a78af0436b122957825ff19bddc`.

## Verdict

The operational home completed a real run against this checkout's default persisted
session store, goal registry, workspace binding, and a short-lived source-built fak
gateway. The connected capture showed five live gateway sessions, four persisted runs,
eight clickable dashboard destinations, a workspace-armed six-tool catalog, and the
declared five-second refresh at one 1440px browser width.

The run also produced the required failing-before evidence. It surfaced three defects,
all filed with stable marker keys:

- [#8562](https://github.com/anthony-chaudhary/fak/issues/8562) — persisted offline proof runs are indistinguishable from live operational runs (`dogfood-harnessweb-offline-run-provenance`).
- [#8563](https://github.com/anthony-chaudhary/fak/issues/8563) — `/api/status` says `mode:"live"` while the configured gateway is unreachable and the browser says offline (`dogfood-harnessweb-status-mode-reachability`).
- [#8564](https://github.com/anthony-chaudhary/fak/issues/8564) — the home does not project the live #8247 intent and DOS lease when goal/gateway state does not carry them (`dogfood-harnessweb-local-work-projection`).

No other defect was observed in this bounded run.

## Real inputs and safety boundary

- The harness read the default user session store. Its four `local-*` rows were real
  persisted results from earlier offline Tool, Approval, and Failure proof scenarios;
  this capture did not seed or rewrite them.
- `fak goal list` returned an empty real registry. No synthetic goal was created.
- The source-built gateway reattached the workstation's durable fak session state and
  exposed five rows through its bounded `/debug/vars` projection during the connected
  capture.
- The gateway was started with `--native --native-code-workspace <repo-root>`. Its
  `/healthz` response reported `armed:true` and Read/Grep/Write/Edit/Bash/Glob. Browser
  status exposed only a stable `ws-...` identity, never the workspace path.
- No model request was sent. The gateway's empty-`--base-url` deterministic planner was
  present but unused; this was a real operational-state read, not a synthetic model turn.
- At capture time `fak intent list` independently showed live target `#8247`, and DOS
  independently showed the exclusive `harnessweb` lease over the three declared trees.
  Their absence from browser state is the evidence for #8564.
- Both loopback processes were stopped after capture, and ports 18080 and 18747 were
  confirmed free.

## Captured states

| State | Gateway | Live sessions | Stored runs | Goals | Dashboard destinations | Clickable links | Workspace | Render |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Failing-before, configured but unreachable | no | 0 | 4 | 0 | 8 | 0 | unarmed | [`dogfood-live-work-offline-1440-2026-08-22.png`](dogfood-live-work-offline-1440-2026-08-22.png) |
| Connected real control plane | yes | 5 | 4 | 0 | 8 | 8 | armed, 6 tools | [`dogfood-live-work-connected-1440-2026-08-22.png`](dogfood-live-work-connected-1440-2026-08-22.png) |

The exact sanitized handler responses are
[`dogfood-live-work-offline-status-2026-08-22.json`](dogfood-live-work-offline-status-2026-08-22.json)
and
[`dogfood-live-work-connected-status-2026-08-22.json`](dogfood-live-work-connected-status-2026-08-22.json).

The browser source schedules `setInterval(refreshOverview,5000)`, satisfying the
five-second refresh ceiling. Playwright used Chrome with a 1440×1650 viewport and a
full-page capture.

## Reproduction

Build the committed snapshot outside the peer-dirty tree, then run the two loopback
processes:

```text
git archive HEAD                         # extract under allocated scratch
go build -o <scratch>/fak-live.exe ./cmd/fak
go build -o <scratch>/harnessweb.exe ./cmd/harnesswebdemo
<scratch>/fak-live.exe serve --addr 127.0.0.1:18080 --native --native-code-workspace <repo-root>
<scratch>/harnessweb.exe -addr 127.0.0.1:18747 -fak-url http://127.0.0.1:18080 -workspace <repo-root>
curl http://127.0.0.1:18747/api/status
playwright screenshot --channel chrome --viewport-size "1440,1650" --full-page http://127.0.0.1:18747/ <output.png>
```

Package and explicit-owned-path validation:

```text
wsl.exe bash -lc 'cd /mnt/c/work/fak && go test ./internal/harnessweb -count=1'
fak validate --wsl-tests --mine docs/_witnesses/harness-web-demo/README.md --mine docs/_witnesses/harness-web-demo/DOGFOOD-LIVE-WORK-2026-08-22.md --mine docs/_witnesses/harness-web-demo/dogfood-live-work-offline-status-2026-08-22.json --mine docs/_witnesses/harness-web-demo/dogfood-live-work-offline-1440-2026-08-22.png --mine docs/_witnesses/harness-web-demo/dogfood-live-work-connected-status-2026-08-22.json --mine docs/_witnesses/harness-web-demo/dogfood-live-work-connected-1440-2026-08-22.png
```

## Artifact hashes

```text
CE0D3B47C53D1C5A55E5CA127323A2FC1EA5AAA095B4CB8237409F1869367962  dogfood-live-work-offline-status-2026-08-22.json
C2E61F38E000901BEC69DAFB0C2E9F0DF3E83C71E1C9B056E21F05A05466F09B  dogfood-live-work-offline-1440-2026-08-22.png
A99A72FD7BE62FFB245B0E4E75545D2FE352163B4C9FDA8A15EBB33FC7C8213F  dogfood-live-work-connected-status-2026-08-22.json
53269159334F18B89A43719D6BEF511408CCD47C66E6B5625C470FE51F101F0F  dogfood-live-work-connected-1440-2026-08-22.png
```

