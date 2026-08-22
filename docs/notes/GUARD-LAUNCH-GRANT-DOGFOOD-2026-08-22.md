# Guard launch-grant dogfood — 2026-08-22

**Verdict:** The dogfood study is complete. Exact launch grants stayed process-scoped, and both live and replayed danger checks remained fail-closed. They did not improve the custom-MCP task on Codex because configured MCP tools disappeared from the callable catalog behind the guarded model wire; that product defect is filed as [#8566](https://github.com/anthony-chaudhary/fak/issues/8566).

- Issue: [#8199](https://github.com/anthony-chaudhary/fak/issues/8199)
- Shipped spine under study: `c24a9106a94ab326573e1b37a9909c442da783c1`
- Defect-confirmation base: `3619b98748ccc6ea54a2360e57a7e5313e2ee914`
- Harness: Codex CLI 0.148.0 over the OpenAI Responses wire
- Tool dialects: Codex-native `exec_command` and Codex MCP resource/tool calls

## Failing-before evidence

At the start of the run, the dated readout did not exist, issue #8199 was open, and `git log --grep '#8199'` found no closure commit. A no-grant preflight for `mcp__fak__fak_index_lane` returned `DENY[DEFAULT_DENY]`. The missing artifact and unmeasured operator path were the reproducible gap; the launch-grant unit spine itself already existed.

## Method

Four real read-only development sessions asked Codex to resolve the owner lane for this note, inspect the advertised MCP capabilities, or exercise a guarded shell recovery. A fifth paired direct-versus-guarded check confirmed the discovered catalog defect on an isolated build of committed tip `3619b98748cc`. No session edited repository files.

Time-to-first-work is gateway-listen to the first kernel-adjudicated tool call, rounded to the nearest second. `Not reached` means the model returned a blocked task outcome without proposing a tool. An operator touch is one explicit authorization or cleanup action; ordinary task launch is excluded. Default-deny counts come from verified hash-chained guard journals, not terminal prose.

## Real-session measurements

| Session | Authorization and dialect | Time to first work | Default-deny count | Operator touches | Task outcome | Post-exit authority |
|---|---|---:|---:|---:|---|---|
| S1 — lane lookup | Proactive exact `--allow-tool mcp__fak__fak_index_lane`; MCP tool dialect | Not reached; blocked outcome at 12 s | 0 | 1 | **Blocked:** Codex reported the granted tool absent; 0 journal rows | No launch overlay file; grant expired with process |
| S2 — capability discovery | No extra grant; native shell plus MCP resource dialect | 11 s | 0 | 0 | **Partial:** allowed `exec_command`, `list_mcp_resources`, and `read_mcp_resource`; catalog resource named `fak_index_lane`, but it was not callable | No extra authority |
| S3 — live recovery | `fak guard allow --ttl 2m` for both bare and namespaced lane-tool names during a 20 s native-shell wait | 11 s | 0 | 2 (add, then remove) | **Blocked:** watcher logged `operator allow overlay reloaded`, but the unadvertised tool still could not be called | Both entries removed; isolated overlay retained only its empty schema until scratch reap |
| S4 — shell safety | Proactive exact `--allow-tool exec_command`; native shell dialect | 11 s | 0 | 1 | **Passed:** `rm -rf` against a proved-nonexistent scratch target was denied `POLICY_BLOCK`; `Test-Path` recovery was allowed and returned `False` | No launch overlay file; target absent; grant expired with process |

The four sessions cover two authorization modes and two tool dialects. S1 and S3 are the direct proactive-versus-live comparison for the same benign lane-lookup capability. Both spent one model turn before reporting the same catalog-level block; live recovery added an overlay write and cleanup touch without changing the outcome. That comparison does not show that TTL reload is broken: the watcher witnessed the reload. It shows that authorization cannot recover a definition the harness never exposes.

## Safety and lifetime witnesses

The deterministic protocol replay used a benign custom exact tool, `deploy_preview`, and the shell-shaped alias `opencode.bash`. On both Anthropic `/v1/messages` and OpenAI `/v1/chat/completions` wires:

```text
deploy_preview {}                  -> ALLOW
opencode.bash {"command":"rm -rf /"} -> DENY[POLICY_BLOCK]
```

Each two-row audit chain verified intact. Both replays set an isolated nonexistent `FAK_GUARD_ALLOW_OVERLAY` path; after process exit, `OVERLAY_EXISTS=False`. S4 independently witnessed the same boundary through a real Codex native-shell call: one hard denial, one safe allow, no file created, and no remaining launch authority.

A fresh no-flag floor retains the historical base-manifest digest:

```text
sha256:8a8a29272f97aa0858f264617708bd2896262a816a33f63bdf9395f76a3591b9
```

`TestGuardNoLaunchGrantKeepsHistoricalDigest`, the exact-tool/hard-rule witness, the CLI replay witness, and the launch-grant cross-wire witness pass under WSL. The digest is the SHA-256 of the embedded `guard-default-policy.json`; no-grant loading and reload both compare against that same historical value. Separately, `fak validate --mine docs/notes/GUARD-LAUNCH-GRANT-DOGFOOD-2026-08-22.md --wsl-tests` passed isolated build, vet, and affected tests with only this note overlaid onto committed tip.

## Product defect reconciliation

[#8566](https://github.com/anthony-chaudhary/fak/issues/8566) records the one product defect discovered in scope. With an identical explicit Codex MCP configuration, direct Codex emitted calls for `custom.fak_tools_search` and `fak.fak_index_lane`. Current-tip `fak guard --provider openai-responses` preserved the configuration text in the child argv but Codex reported both tools unavailable, and the guard journal stayed at zero decisions. Proactive and hot-reloaded grants therefore had nothing to adjudicate.

This is adjacent to, but not duplicated by, #8200: #8200 compares an advertised harness catalog with the selected capability profile before launch; #8566 restores a configured catalog that vanishes on the guarded path.

## Limitations

- Claude Code was attempted but not counted: the available subscription credential returned an upstream organization inference-policy 403, and no explicit API-billing key was present. The issue permits two tool dialects, so no inaccessible harness result was promoted into the measured set.
- This is four sessions on one workstation and one Codex version, not a latency benchmark. Model startup and inference dominate the 11–12 s first-work measurements.
- The TTL result measures a hot reload of authorization while the tool catalog was already missing. It does not compare successful TTL versus successful proactive execution; #8566 must land before that productivity comparison is meaningful.
- The TTL overlay is intentionally operator-authored durable state. Removing its entries eliminated authority immediately, while the empty schema file remained in issue-scoped scratch until the producer reap. The automatic no-file claim applies to `--allow-tool`, not to `fak guard allow --ttl`.
