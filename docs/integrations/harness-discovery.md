---
title: "Scoped harness discovery"
description: "fak harness discover turns the explicit selection manifest into a provenance-bearing candidate set before composition:"
---
# Scoped harness discovery


`fak harness discover` turns the explicit selection manifest into a provenance-bearing candidate set before composition:

```text
fak harness discover --registry ~/.config/fak/harness-registry.json --path ./current-project --principal engineer@example.com
fak harness select   --discover ~/.config/fak/harness-registry.json --path ./current-project --principal engineer@example.com --tag legal
```

Registry schema `fak.harness-discovery/v1alpha1` declares company, team, person, and project sources. With `discover_repo: true`, fak also walks upward from `--path` for `.fak/harness.json`. Discovery order does not determine precedence; it emits one `fak.harness-selection/v1alpha1` manifest for the contextual resolver.

Each candidate reports scope, owner, canonical source path, SHA-256 content digest, trust class, signer, refresh policy, and manifest schema. Company/team managed declarations must carry a detached Ed25519 signature from `trusted_signers`. Team/person declarations must admit the authenticated `--principal`. Paths are relative to a declared root, resolved through symlinks, and rejected if they escape that root. Duplicate scope/ID identities, unreadable declarations, revoked digests, invalid signatures, and scope-mismatched layers fail before selection.

Refresh policies are deliberately descriptive and closed: `immutable`, `session`, or `manual`. This spine rereads sources on every invocation and does not create an offline cache, so stale cache and revocation races cannot yet occur. A future cache must key by content digest, re-check the registry revocation set before use, and record its refresh receipt; it must not silently copy declarations into Codex, Claude, Copilot, or other host config directories.

Discovery still stops short of domain/task inference (#6900), typed asset composition (#6792/#6904), and external launch integration (#6901). The explicit registry is the current trust and identity boundary, not a claim that ambient OS identity alone authenticates an engineer.

## Witness

Issue #6898 is covered by `internal/harnessdiscover` adversarial tests and `cmd/fak/TestHarnessDiscoverAndSelectCLI`. The isolated CLI witness passed in 0.01s (`ok command-line-arguments 0.019s`) against archived trunk plus only the declared paths.
