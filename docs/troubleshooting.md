---
title: "fak troubleshooting: start from the user-visible symptom"
description: "Current recovery route for fak users: capture the symptom, choose the matching check, and continue to the maintained server diagnosis."
---

# Troubleshooting route

This page is for **users whose `fak serve` gateway did not start or stopped
behaving as expected**. Begin with the visible symptom and the least-mutating
check; contributor test failures and implementation debugging belong on deeper
routes.

**Next action:** ask the deployed gateway whether it is serving:

```bash
curl -s http://127.0.0.1:8080/healthz
```

Change the host and port to the deployed address. JSON with `"ok":true` means the
HTTP process is responding; a connection error or unhealthy response selects the
startup path below. This check does not prove model quality or end-to-end task
success.

## Choose the visible symptom

| What the user sees | First check | Continue here |
|---|---|---|
| **fak refused to start and printed a `reason:` token.** | Run the `next:` command the block already printed; it names the checks for that exact token. | [Config bails](config-bails.md) |
| **The address is already in use.** | Identify the process that owns the configured port; otherwise choose another explicit `--addr`. | [Port conflicts](fak/server-troubleshooting.md#port-conflicts) |
| **The process exits or never becomes healthy.** | Read the first startup error and confirm the address, engine/model source, policy path, and authentication environment. | [Startup failures](fak/server-troubleshooting.md#startup-failures) |
| **Every outbound call fails on a certificate the host cannot verify** (`x509: certificate signed by unknown authority`, `SELF_SIGNED_CERT_IN_CHAIN`, `CRYPT_E_NO_REVOCATION_CHECK`) — and the same call fails from `curl` too. | Run `fak doctor trust`; it names the intercepting CA, the trust source in use, and the child runtimes that would not inherit it, read-only. | [Managed hosts](managed-hosts.md) |
| **`fak guard` waits for a login on a host whose cloud credential already works** (Bedrock/Vertex). | Run `fak doctor trust`; a `cloud-route` finding means the child signs its own requests and ignores `ANTHROPIC_BASE_URL`. | [Managed hosts](managed-hosts.md#symptom-2-a-24-hour-wait-for-a-login-that-cannot-happen) |
| **Model loading reports missing, invalid, or tokenizer data.** | Confirm the configured GGUF/model path exists and matches the selected serving mode. | [Model loading failures](fak/server-troubleshooting.md#model-loading-failures) |
| **Loading fails with out-of-memory.** | Compare the selected model and context size with available memory before retrying. | [Memory issues](fak/server-troubleshooting.md#memory-issues) |
| **CUDA, Vulkan, or another accelerator fails.** | Capture the exact device/runtime error and verify the selected backend on the machine that owns the hardware. | [GPU/CUDA issues](fak/server-troubleshooting.md#gpucuda-issues) |
| **Policy validation or authentication fails.** | Validate the policy with `fak policy --check policy.json`; then confirm the named key environment variable exists in the service environment. | [Policy and configuration](fak/server-troubleshooting.md#policy-and-configuration-issues) |
| **The gateway is healthy but requests are slow, denied, or failing.** | Correlate `/metrics`, structured logs, and `/debug/vars`; use the request `trace_id` for one-request diagnosis. | [Observability question router](observability/README.md) |
| **The symptom began after an upgrade.** | Preserve the failing revision and evidence, then compare with the last known release or promoted stable anchor. | [Operator recovery and upgrade order](operator/README.md#recovery-and-upgrade-order) |

## Recovery order

1. Capture the exact error, HTTP status, `trace_id`, and deployed revision before
   changing the process.
2. Run the symptom's first check and follow the linked maintained diagnosis.
3. Change one address, model, policy, backend, credential, or release input at a
   time.
4. Re-run `/healthz`, then repeat the request or telemetry check that exposed the
   symptom.

This order separates observation from mutation. It also leaves a before/after
witness when the recovery works.

## User route versus contributor diagnostics

The user route ends when the documented configuration or runtime check restores
the service, or when the symptom is captured well enough to escalate. Commands
for repository tests, build tags, fixture traces, engine development, or source
instrumentation are contributor diagnostics; use them only when reproducing the
problem in a source checkout. They are not prerequisites for operating an
installed gateway.

If the detailed server guide does not cover the captured symptom, report the
binary revision, platform, serving mode, redacted command/configuration, exact
error, and relevant health/metrics/log excerpt. Do not include API keys, request
bodies, tool arguments, or result content.

## Mode, generation, lifecycle, and support

| Context | Meaning for this route |
|---|---|
| **Mode** | The default path is a networked `fak serve` gateway. Offline `preflight`/agent proofs and contributor tests have different failure surfaces. |
| **Generation** | This is the current `gen/now` symptom router; the linked server troubleshooting guide owns detailed shipped diagnoses. |
| **Lifecycle** | Capture evidence, diagnose, change one input, verify, then either return to service or escalate the reproducible symptom. |
| **Support** | Documented server flags, endpoints, policies, backends, install/release channels, and their linked runbooks define the supported surface. Private infrastructure and contributor internals remain on their scoped routes. |
| **Runtime authority** | The deployed binary's `fak serve --help`, configuration environment, endpoint responses, and emitted errors determine actual behavior at that revision. |

For the full production sequence, return to the [operator route](operator/README.md).
For detailed signal selection, use the [observability route](observability/README.md).
