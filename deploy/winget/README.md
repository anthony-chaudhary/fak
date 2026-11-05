---
title: "fak on winget — committed manifest set"
description: "A committed winget multi-file manifest (version + installer + locale) for the published windows/amd64 release asset, ready for a one-time microsoft/winget-pkgs submission."
---

# fak on winget

Windows is a first-class published target (`release-artifacts.yml` builds
`windows/amd64` on every release), and winget is the top Windows deployment
option — but a package only exists in winget once a manifest is submitted to
[microsoft/winget-pkgs](https://github.com/microsoft/winget-pkgs). This
directory commits that manifest so the submission is a copy-and-verify step
instead of a from-scratch authoring job. The
[deployment guide](../../docs/fak/deployment-guide.md) and
[INSTALL.md](../../INSTALL.md) remain the source of truth for what the binary
does once installed.

## What's here

A **multi-file manifest** (schema `1.6.0`), the form `wingetcreate` and the
winget-pkgs PR pipeline expect:

| File | ManifestType | What it is |
|---|---|---|
| [`AnthonyChaudhary.Fak.yaml`](AnthonyChaudhary.Fak.yaml) | `version` | Identifier + version + default locale pointer. |
| [`AnthonyChaudhary.Fak.installer.yaml`](AnthonyChaudhary.Fak.installer.yaml) | `installer` | The `windows/amd64` zip: real release URL, real SHA-256, `NestedInstallerType: portable` on `fak.exe`. |
| [`AnthonyChaudhary.Fak.locale.en-US.yaml`](AnthonyChaudhary.Fak.locale.en-US.yaml) | `defaultLocale` | Publisher, license (Apache-2.0), description, tags. |

## PackageIdentifier

`AnthonyChaudhary.Fak` — winget identifiers are `Publisher.Package`, derived from
the GitHub publisher (`anthony-chaudhary`). This is the intended identifier for
the submission; if it collides at submission time, wingetcreate's validation will
say so and the prefix changes in exactly these three files. After submission,
users install with:

```powershell
winget install AnthonyChaudhary.Fak
```

## Provenance of the checksum (real, not placeholder)

- **Asset name/format** comes from `.github/workflows/release-artifacts.yml`:
  `fak_<version>_windows_amd64.zip`, containing `fak.exe`, `LICENSE`, and
  `GETTING-STARTED.md` at the archive root.
- **`InstallerSha256` is REAL** for `fak_0.56.0_windows_amd64.zip`
  (`B1471096EB41C70476CF57F84BC0EF959223A84B26DA041260358DF9C5F2FFE7`), taken
  from the GitHub release-asset digest (releases API) for the v0.56.0 asset.
  Cross-check it against the release's `SHA256SUMS` asset before submitting:
  `Select-String " windows_amd64.zip$" SHA256SUMS`.
- **Version**: `0.56.0` is the latest GitHub release as of this commit.
  `Formula/fak.rb` (the Homebrew formula) may lag the latest release; the winget
  manifest should always track the **latest release**, not the formula.

## Refreshing per release

Every release publishes a new `fak_<version>_windows_amd64.zip` with a new
checksum. To refresh: copy this directory, bump `PackageVersion` in all three
files, point `InstallerUrl` at the new asset, and replace `InstallerSha256` with
the new asset's digest (releases API or `SHA256SUMS`). Then submit (below).

## The one-time submission step (NOT performed from this repo)

Submission to `microsoft/winget-pkgs` is a deliberate, human-approved action —
do it from a maintainer machine, not from CI:

1. **Validate locally** (optional but recommended):
   `winget validate --manifest deploy/winget` (or point `wingetcreate validate`
   at the directory).
2. **Submit with wingetcreate** (opens the PR for you):
   `wingetcreate update AnthonyChaudhary.Fak --version 0.56.0 --urls <windows-amd64-zip-url> --submit`
   — or submit the manifest files as-is with
   `wingetcreate submit deploy/winget`.
3. **Or open a PR by hand** against `microsoft/winget-pkgs` with the files under
   `manifests/a/AnthonyChaudhary/Fak/0.56.0/`, matching the repo's folder
   convention.

After the PR merges, `winget install AnthonyChaudhary.Fak` resolves. Subsequent
releases are the same submission with a new version — hence "one-time" only for
the first publication.
