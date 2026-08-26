# Migrate a job-apply app to fak in 15 minutes

This runbook is the shortest supported path for a **signed desktop app that already sends a job-application prompt to a model**. It replaces that transport with the bundled fak helper while leaving the app's forms and rendering code intact. The user installs one app bundle: they do **not** install Ollama or LM Studio and do not run terminal setup.

The stable protocol and security boundaries are defined in [Local-app compute layer](local-app-compute-layer.md). Names such as `FakKit` below describe the thin host-app adapter you own; they are integration pseudocode, not a claim that fak ships an Apple framework or package manager artifact.

## The 15-minute day-one migration

### Minute 0-3: bundle the helper

Build the pinned `fak` source or copy the release binary into the app's signed helper location (`Contents/Helpers/fak` on macOS). Record its SHA-256 in the app release manifest. Sign the helper with the same team identity and hardened-runtime policy as the host, then sign the outer bundle and submit the complete bundle for notarization. CI must verify the helper hash, nested signature, outer signature, and notarization ticket before publishing.

Do not download or replace the executable after signing. The host launches only its bundled absolute path; it never resolves `fak` through `PATH`. On Windows use the equivalent signed application directory and Authenticode verification. On Linux verify the package signature and immutable packaged path.

### Minute 3-6: add the FakKit adapter and declarations

Add a small `FakKit` adapter in the host process. It owns helper lifecycle, framed stdin/stdout, event decoding, cancellation, and receipt persistence. It must not own policy decisions or parse prose as protocol.

Declare the task and pack channel in app-owned configuration:

```json fak-localapp-declaration
{
  "protocol": "fak-local-app/1",
  "tasks": ["job.apply"],
  "pack_channel": "stable",
  "helper_path": "Contents/Helpers/fak",
  "receipt_policy": "scrubbed"
}
```

`job.apply` is this app's declaration, not a built-in fak entitlement. The helper must reject undeclared task IDs and unknown pack channels. Pin the protocol, helper version, and pack digest in production.

### Minute 6-10: replace only the model-call transport

Keep the existing `ApplicationDraft` input and output types. Replace the network SDK call with one framed request to the bundled helper. Illustrative host pseudocode:

```swift
let request = FakRequest(task: "job.apply", input: draft.redactedForLocalCompute())
for event in try await fakKit.run(request) {
    switch event {
    case .ready:                 view.showRunning()
    case .output(let result):    view.render(result.structured)
    case .handoffRequired(let h): view.showExplicitHandoff(h)
    case .failed(let error):     view.showRecoverableFailure(error)
    }
}
```

The app sends one length-delimited request and accepts only typed envelopes. It renders a structured result only after schema validation. It never treats logs, stderr, or arbitrary text as output.

### Minute 10-13: map events to user states

| Helper event | Required user state | Forbidden shortcut |
|---|---|---|
| startup, before `ready` | **Starting local compute…** with cancel | claim that work is running |
| `ready` | **Working locally…** | expose helper logs as UI |
| `output` with valid schema | reviewable structured application | silently submit externally |
| `handoff_required` | explicit reason, destination class, and Continue/Cancel | automatic cloud fallback |
| typed failure | retry/support choices and stable error code | parse prose or retry forever |
| cancellation acknowledged | idle, no partial result | retain an active-looking spinner |

`handoff_required` is an outcome, not an error. Continue is a fresh, explicit user decision. The helper must never silently route to a remote model.

### Minute 13-15: run the clone-to-result witness

From a clean clone, run:

```console
$ go test ./cmd/fak -run TestLocalAppJobApplyRunbookWitness -count=1 -v
```

The test extracts and executes both fixtures below. It captures one local structured result and one forced explicit handoff; it requires no model, key, GPU, Ollama, or LM Studio.

```json fak-localapp-fixture
{
  "schema": "fak-local-app-witness/1",
  "cases": [
    {
      "name": "local-structured-result",
      "request": {"task":"job.apply","input":{"role":"support engineer","skills":["Go","incident response"]}},
      "force_handoff": false,
      "expected_events": ["ready","output"],
      "expected_result": {"status":"drafted","sections":["summary","skills"]}
    },
    {
      "name": "forced-explicit-handoff",
      "request": {"task":"job.apply","input":{"role":"regulated reviewer","skills":["audit"]}},
      "force_handoff": true,
      "expected_events": ["ready","handoff_required"],
      "expected_handoff": {"explicit":true,"reason":"task_requires_remote_capability"}
    }
  ]
}
```

Passing means the documentation artifact is executable and the event/UI contract is intact. It does not certify a particular model's writing quality or platform code-signing identity.

## Day one versus production adoption

| Concern | Day one (the 15-minute diff) | Production requirement |
|---|---|---|
| Packaging | bundled helper at an absolute path | reproducible pinned build, hash manifest, nested signing, notarization/package verification |
| Adapter | launch, frame, decode, cancel | supervised lifecycle, bounded restart, backpressure, version negotiation |
| Tasks/packs | declare `job.apply` and `stable` | signed allowlist, pinned pack digest, staged channel promotion |
| Transport | replace the existing model SDK call | schema/version compatibility tests and downgrade refusal |
| UX | starting, working, result, handoff, failure | accessibility, localization, cancellation and crash recovery |
| Receipts | scrubbed receipt saved locally | retention controls, export/delete UI, support consent, tamper evidence |
| Updates | manual app update | atomic host+helper compatibility gate and staged rollout |
| Support | show stable code and versions | one-click scrubbed diagnostic bundle and documented escalation |

## Receipts and privacy

Persist a scrubbed receipt containing: protocol version, task ID, pack channel/digest, helper version/hash, start/end timestamps, terminal event, schema version, stable error code, and a correlation ID. Exclude prompt text, resume/application content, secrets, environment variables, usernames, raw paths, helper stdout/stderr, and model output. Show the exact bundle to the user before export; support receives it only after consent.

## Updates, rollback, and compatibility

Ship host, helper, declarations, and packs as one compatibility set. Before activation, stage the update, verify signatures/hashes, launch a readiness probe, and verify protocol/task/pack compatibility. Activate atomically only after all checks pass.

Keep the last known-good set. On failed readiness or compatibility, leave the current set active. After activation, rollback is an explicit app action that restores the entire prior set—not only the helper. Never reinterpret a newer request with an older schema, and never recover by searching `PATH`, downloading an unsigned helper, or silently using a remote model.

## Support diagnostics

1. Ask the user for the visible stable error code and whether the terminal event was failure or `handoff_required`.
2. In the app's Support screen, run **Check local compute**. It verifies packaged path, helper hash/signature, protocol version, declared `job.apply`, pack channel/digest, readiness, writable receipt storage, and cancellation.
3. Offer **Preview diagnostic bundle**. Confirm it contains only the scrubbed receipt fields listed above.
4. With consent, export that bundle. Reproduce against the same compatibility set with the witness command.
5. If readiness fails after an update, use **Rollback local compute**. If the prior set also fails, stop retrying and escalate with the stable code and scrubbed bundle.

Support must never request resumes, prompts, secrets, raw logs, or a terminal session from the end user.

## Uninstall

Quit the app, stop the supervised helper, delete queued requests and app-owned receipts according to the user's retention choice, then remove the app bundle/package. The helper, packs, and adapter are inside the app-owned installation and leave with it. Remove any app-created launch registration using the platform uninstaller. Do not remove shared runtimes: this integration installs no Ollama or LM Studio service and changes no shell profile or system `PATH`.

## Supported and unsupported

| Supported | Unsupported |
|---|---|
| Signed/notarized or package-verified host with a bundled pinned helper | helper discovered through `PATH` or downloaded after signing |
| Declared typed tasks and pinned, signed pack channels | arbitrary shell/tool access or undeclared tasks |
| Framed local IPC, typed events, schema-validated output | scraping stdout/stderr or parsing prose as control data |
| Explicit visible handoff with Continue/Cancel | silent cloud/model fallback |
| Scrubbed, consented support receipts | prompts, resumes, secrets, raw logs, or model output in diagnostics |
| Atomic compatibility-set update and whole-set rollback | independent helper/schema updates without compatibility checks |
| App-owned uninstall with no external runtime | requiring Ollama, LM Studio, end-user terminal setup, or shell-profile changes |

## Release checklist

- [ ] Clean-clone witness passes.
- [ ] Helper hash, nested signature, outer signature, and notarization/package verification pass.
- [ ] `job.apply`, protocol, pack channel, and digest are pinned and allowlisted.
- [ ] Structured output and forced explicit handoff render correctly.
- [ ] Diagnostic preview is scrubbed; export requires consent.
- [ ] Failed update preserves or restores the full last-known-good set.
- [ ] Uninstall removes app-owned helper/packs/receipts and no shared runtime.

