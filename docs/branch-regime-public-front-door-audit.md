---
title: "Branch regime public front-door audit"
description: "Classification map for public docs, install links, and contributor branch instructions during the dev/main branch-role migration."
---

# Branch Regime Public Front-Door Audit

**Issue:** #1700.
**Status:** pre-cutover audit. Public/adopter surfaces remain anchored on
`main` and release artifacts; contributor/agent surfaces keep the current
`main` operator law until #1703 records a clean shadow cutover and the guards,
workflows, and prompts can move together.

This is not a `main` -> `dev` replacement list. The branch regime intentionally
keeps two meanings alive:

- public front door: stable user-facing browsing, install, Pages, badges,
  examples, and release pages stay on `main` / release tags.
- contributor workflow: everyday work currently lands on `main`; after cutover,
  those instructions move to the configured development branch in one coordinated
  prompt/hook/docs change.

## Regeneration Commands

```bash
rg -n "github\\.com/anthony-chaudhary/fak/(blob|tree)/main|raw\\.githubusercontent\\.com/anthony-chaudhary/fak/main|badge\\.svg\\?branch=main|go install github\\.com/anthony-chaudhary/fak/cmd/fak@(latest|main|master)|git (fetch|merge|push) origin[/ ](main|master)|origin/(main|master)" README.md INSTALL.md GETTING-STARTED.md START-HERE.md AGENTS.md CONTRIBUTING.md .github/copilot-instructions.md docs/integrations docs/fak/deployment-guide.md
go test ./internal/branchrole -run TestPublicDocRefsCurrentTreeClassified -count=1
```

The Go gate is narrow by design: it scans branch-shaped public URLs, install
commands, and explicit contributor branch commands. It does not treat ordinary
prose like "main agent" or third-party asset paths as branch-regime evidence.

## Public Front Door - Keep On `main`

| Surface | Current shape | Classification | Why it stays |
|---|---|---|---|
| `README.md` workflow badges | `badge.svg?branch=main` | public front door | The README is the public release branch's first screen; these badges should describe the front-door branch until a separate dev badge is deliberately added. |
| `README.md` Colab links | `colab.research.google.com/.../blob/main/...` | public front door | A cold adopter should open the stable notebook, not the hot integration branch. |
| `README.md`, `INSTALL.md`, `GETTING-STARTED.md`, integration guides, deployment guide | `raw.githubusercontent.com/anthony-chaudhary/fak/main/install.sh` | public front door | The installer script is the public on-ramp. It resolves the released binary by default and can be pinned with `FAK_VERSION`. |
| Integration and deployment docs | `github.com/anthony-chaudhary/fak/blob/main/...` / `tree/main/...` | public front door | Public docs should link to browseable stable examples and policy manifests. |

## Release Artifacts - Keep On Tags / `@latest`

| Surface | Current shape | Classification | Why it stays |
|---|---|---|---|
| `README.md`, `AGENTS.md`, `.github/copilot-instructions.md`, `GETTING-STARTED.md`, `INSTALL.md`, adopter docs | `go install github.com/anthony-chaudhary/fak/cmd/fak@latest` | release artifact | `@latest` resolves the newest module tag. It should not become `@dev`, `@main`, or `@master`. |
| Install and deployment docs | `github.com/anthony-chaudhary/fak/releases/latest` | release artifact | Manual binary downloads should land on the released asset page, not on a branch ref. |

## Contributor Workflow - Keep Current Law Until Cutover

| Surface | Current shape | Classification | Required move |
|---|---|---|---|
| `AGENTS.md` | work directly on `main`, fetch/merge `origin/main`, push to `main` | contributor workflow | Keep until #1703 proceeds. Then replace with configured development-branch text in the same change that updates hooks, workflows, dispatch prompts, and recovery text. |
| `CONTRIBUTING.md` | work directly on `main`, no feature branch | contributor workflow | Same as above. This is correct pre-cutover and wrong only after the guard accepts `dev`. |
| `.github/copilot-instructions.md` | work directly on the trunk (`main`) | contributor workflow | Same as above; it must not move ahead of the authoritative agent prompt. |

## Adjudication Fixtures - Keep Literal Hazard Commands

| Surface | Current shape | Classification | Why it stays |
|---|---|---|---|
| `docs/fak/deployment-guide.md` | JSON payload containing `git push origin main` | adjudication fixture | It is a dangerous-command example used to prove the gateway refuses the operation, not a workflow instruction. |
| `docs/integrations/claude.md` and similar integration docs | policy tables with `git push origin master` / branch push examples | adjudication fixture | These rows demonstrate refusal behavior. They are not branch-role guidance. |

## Legacy Residue

No public-front-door `main` link is currently classified as legacy residue by
this issue. Historical or fixture-only `main`/`master` references outside the
public/contributor surfaces are tracked by
[`docs/branch-regime-hardcoded-ref-audit.md`](branch-regime-hardcoded-ref-audit.md).

## Guardrail

`internal/branchrole` owns the regression gate:

- `TestPublicDocRefsCurrentTreeClassified` fails any new unclassified
  branch-shaped public/contributor doc reference.
- `go install ...@latest` is classified as a release artifact.
- `go install ...@main` and `go install ...@master` are intentionally
  unclassified so a mutable public install command reds the package.

## Cutover Rule

Do not update `AGENTS.md`, `CONTRIBUTING.md`, or
`.github/copilot-instructions.md` to say everyday work targets `dev` until the
shadow cutover checklist says `proceed`. Public install and adopter links should
remain on `main` / release tags after that cutover; only contributor workflow
instructions move to the configured development branch.
