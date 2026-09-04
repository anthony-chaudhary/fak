---
title: "fak releases: current install and upgrade authority"
description: "Operator index that routes current fak installation and upgrade work before versioned release history and stable rollback evidence."
---

# Releases

This page is for **operators choosing, installing, or reviewing a released `fak`
version**. Current actions come first; files in this directory are version-scoped
release records, not evergreen installation instructions.

**Next action:** identify the version and provenance of the binary you are running:

```bash
fak version
```

A release version identifies the artifact generation. Retain the build provenance
or artifact checksum too; output that says no VCS stamp cannot identify a source
commit by itself.

## Current release actions

| Operator task | Current authority | What it decides |
|---|---|---|
| **Install fak** | [Install-path chooser](../adoption/install-paths.md) | Select a prebuilt artifact, `go install`, source build, or MCP setup. Unpinned `latest` resolution happens when the command runs; pin a tag when reproducibility matters. |
| **Upgrade or roll back a deployment** | [Upgrade route](../upgrade.md) | Record source and target versions, validate configuration and consumers, stage and verify the target, and retain the prior artifact or revision for rollback. |
| **Choose a stable rollback anchor** | [Stable-release authority](../stable-releases/README.md) | A `stable/<codename>` tag promotes an already-shipped rolling commit after evidence and soak; it is a commit anchor, not a new artifact stream. |
| **Download the current published artifact** | [GitHub releases](https://github.com/anthony-chaudhary/fak/releases/latest) | The latest published rolling `vX.Y.Z` release and its platform assets/checksums. Record the resolved tag before deployment. |
| **Track in-flight work in progress** | [`NEXT.md`](NEXT.md) | Living draft tracking unreleased commits on `main` targeting the upcoming release. Inspected with `fak release next` and synced with `fak release next --sync`. |

Rolling releases are the active binary distribution. Stable tags are sparse
rollback anchors over commits already shipped by the rolling channel. An
untagged trunk checkout is source development, not a published release.

## The "Next" version concept (proactive WIP tracking)

Between tagged releases, all work in progress on `main` belongs to the "Next"
version (`vNext`). Rather than compiling release notes at the moment of the cut,
in-flight features, bug fixes, and upgrade guidance are proactively tracked in
[`NEXT.md`](NEXT.md).

- **Inspect in-flight changes:** run `fak release next` (or `--json`) to view base tag,
  commits in flight, projected semver bump level, and sync status.
- **Sync the living draft:** run `fak release next --sync` to update [`NEXT.md`](NEXT.md)
  with newly landed commits while preserving custom highlights and upgrade notes.
- **Release cut promotion:** when running `fak release ship` or `fak release cut`,
  the curated notes from [`NEXT.md`](NEXT.md) are automatically promoted into the
  versioned record `vX.Y.Z.md`, and [`NEXT.md`](NEXT.md) is reset for the next cycle.

## Versioned release records

The `vX.Y.Z.md` files beside this README describe their named release. Use one
when evaluating or reproducing that exact target; newer current guidance lives in
the authorities above.

| Recent record | Scope |
|---|---|
| [`v0.41.0.md`](v0.41.0.md) | Evidence and notes for rolling release `v0.41.0`. |
| [`v0.40.0.md`](v0.40.0.md) | Evidence and notes for rolling release `v0.40.0`. |
| [`v0.39.0.md`](v0.39.0.md) | Evidence and notes for rolling release `v0.39.0`. |
| [`v0.38.0.md`](v0.38.0.md) | Evidence and notes for rolling release `v0.38.0`. |
| [`v0.37.0.md`](v0.37.0.md) | Evidence and notes for rolling release `v0.37.0`. |
| [`v0.36.0.md`](v0.36.0.md) | Evidence and notes for rolling release `v0.36.0`. |

Older `v*.md` records remain available in this directory by exact version. A
filename containing `candidate`, `strategy`, or an issue number is planning or
historical context unless a current authority explicitly promotes it. It does
not override the tagged release, its published assets, or current operator
routes.

## Which evidence answers which question?

- **What can I install now?** Read the latest published GitHub release, then use
  the install-path chooser.
- **What changed in one target?** Read that exact `vX.Y.Z.md` record and the
  published release entry.
- **How do I move production safely?** Use the upgrade route; a release note does
  not replace staging, configuration validation, health, behavior, and telemetry
  checks.
- **Which known commit can I restore?** Use the stable-release evidence and your
  retained artifact/revision. Stable promotion does not manufacture a binary.
- **How are releases produced?** That is a contributor/release-engineering
  workflow, not an operator prerequisite. Current operator authority remains the
  installed artifact, upgrade route, and published evidence.

## Mode, generation, lifecycle, and support

| Context | Meaning for this index |
|---|---|
| **Mode** | Applies to published binaries and explicit source checkouts. Follow the install/upgrade row matching the deployed mode; do not mix artifact and source instructions. |
| **Generation** | This README is the current `gen/now` release index. Rolling `vX.Y.Z` entries are version-scoped records; `stable/<codename>` evidence records sparse promoted anchors; candidate/strategy notes are not current release authority. |
| **Lifecycle** | Publish rolling artifact → retain versioned record → optionally soak/promote its commit as a stable anchor → keep both as historical evidence when newer rolling releases ship. |
| **Support** | GitHub release assets/checksums, install and upgrade routes, exact version records, and stable evidence are scoped authorities. Planning notes and chat/release-channel coordination are not deployment contracts. |
| **Runtime authority** | The selected binary's `fak version`, target `fak serve --help`, policy/configuration validation, and runtime witnesses determine behavior for that release. |

Return to the [operator route](../operator/README.md) for deploy, observe,
recover, and upgrade sequencing.
