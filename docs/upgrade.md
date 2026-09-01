---
title: "Upgrade fak: choose a release, migrate configuration, preserve rollback"
description: "Current operator route for recording a fak source version, choosing a rolling or stable target, validating configuration, and rolling back by artifact or commit."
---

# Upgrade route

This page is for **operators replacing a deployed `fak` binary or checked-out
revision**. An upgrade is an artifact/revision replacement followed by
configuration and service verification; there is no documented in-process
self-upgrade command.

**Next action:** record the version of the binary currently serving traffic:

```bash
fak version
```

Keep that output with the deployed command, policy/configuration, platform, and
health evidence. If the build reports no VCS stamp, record the installation
source or artifact checksum as well; a semantic version without provenance does
not identify a source commit.

## Choose the applicable release generation

| Need | Target | Range and lifecycle | Installation authority |
|---|---|---|---|
| **Current published release** | A rolling `vX.Y.Z` GitHub release | Upgrade from the recorded source to one explicit target tag. `@latest` and an unpinned installer resolve at execution time, so record the resolved version before rollout. | [Install-path chooser](adoption/install-paths.md) |
| **Reproducible published release** | A pinned rolling tag such as `vX.Y.Z` | Pin the exact target for staging and production. On the shell installer, set `FAK_VERSION` without the leading `v`; with Go, use `go install github.com/anthony-chaudhary/fak/cmd/fak@vX.Y.Z`. | [`install.sh` contract](../install.sh) and [install paths](adoption/install-paths.md) |
| **Evidence-backed rollback anchor** | A promoted `stable/<codename>` tag | Stable tags identify previously shipped commits after a soak/gate; they do not create a new binary or imply that they are newer than every rolling release. Source operators can check out the tag; binary operators retain or recover the rolling artifact that points to the same commit. | [Stable-release authority](https://github.com/anthony-chaudhary/fak/blob/main/docs/stable-releases/README.md) |
| **Source checkout** | An explicit commit or tag | The applicable range is exactly the source and target revisions you name. Build and verify that checkout; do not describe an untagged trunk build as a published release. | [From-source install path](adoption/install-paths.md) |

The project does not publish a blanket compatibility range that makes every
configuration valid across every pair of versions. The upgrade range is the
recorded source plus the explicit target. Review the target release evidence and
validate the interfaces your deployment actually consumes.

## Migration checklist

Before replacing the running binary:

1. Record the source version/provenance, target tag or commit, platform, artifact
   checksum when available, and the prior artifact or checkout used for rollback.
2. Compare the target's `fak serve --help` with the deployed command and service
   environment. Reconcile renamed, removed, or newly required flags explicitly.
3. Validate the deployed policy with the target binary:

   ```bash
   fak policy --check policy.json
   ```

4. Recheck provider/model routing, authentication environment, listener address,
   endpoint consumers, metrics/log fields, and any workflow that parses `--json`.
5. Stage the target with production-equivalent configuration, then capture
   `/healthz`, representative request behavior, and the telemetry used by the
   deployment before shifting traffic.

No data migration is implied merely by replacing this static binary. If a target
release introduces a state or schema migration, its release evidence must name
that migration; otherwise do not invent one. Configuration and consumer review
still applies because command and JSON contracts can change independently of
persistent data.

## Roll out and verify

Replace or restart through the deployment's own process manager; `fak` does not
own that orchestration. After startup:

1. Run `fak version` on the target binary and retain its provenance.
2. Require `/healthz` to return `"ok":true`.
3. Verify a representative allowed request and a representative denied tool call.
4. Compare `/metrics`, structured logs, and any `--json` consumer with the staged
   baseline.
5. Keep the previous artifact/revision and configuration until the soak window
   and operator acceptance criteria close.

Use the [observability route](observability/README.md) to choose each production
signal and the [troubleshooting route](troubleshooting.md) when verification fails.

## Rollback

Rollback means restoring the recorded prior artifact or source revision with its
matching configuration, then repeating the same health, behavior, and telemetry
checks. For a source deployment, a promoted stable anchor can be selected with
`git checkout stable/<codename>`. A stable tag is a commit anchor, not an
in-process rollback command and not a replacement for retaining deployable
artifacts.

If the target partially changed an external provider, workflow, secret, or
configuration contract, restore that consumer in lockstep. Preserve the failed
target's redacted evidence for diagnosis rather than deleting the only repro.

## Mode, generation, lifecycle, and support

| Context | Meaning for this route |
|---|---|
| **Mode** | This route covers installed binaries and explicit source checkouts. Select the row matching how the deployment was installed; do not mix binary and source rollback instructions. |
| **Generation** | This is the current `gen/now` upgrade route. Rolling `vX.Y.Z` releases are the current artifact stream; `stable/<codename>` tags are sparse historical rollback anchors with committed evidence. |
| **Lifecycle** | Record → select and pin → validate/migrate → stage → roll out → verify/soak → retire the prior artifact, or roll back. |
| **Support** | Install scripts, tagged release assets, documented CLI/configuration surfaces, and stable-release evidence are supported authorities. A release-specific migration exists only when its evidence names one. |
| **Runtime authority** | The exact target binary's `fak version`, `fak serve --help`, policy validation, endpoint responses, and emitted telemetry determine behavior for that rollout. |

Return to the [operator route](operator/README.md) for the complete deploy,
observe, recover, and upgrade loop.
