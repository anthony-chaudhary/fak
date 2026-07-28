---
title: "Extension descriptor and local conformance"
description: "fak-extension-descriptor/1 is inert discovery metadata; local Verify re-hashes the artifact and re-runs the bounded witness, so catalog data is never the proof."
---

# Extension descriptor and local conformance

`internal/market` defines `fak-extension-descriptor/1`, inert discovery metadata shared by the frozen ABI registry, compute backends, TUI panes, quality checks, and trajectory scorers. A descriptor carries a namespaced identity, `module@rev`, seam ABI range, artifact digest, trust class, error behavior, requested capabilities, and an optional required witness recipe/result digest.

Parsing is deliberately non-executable: catalog enumeration does not import an artifact, call registration, or run a witness. Local `Verify` is the authority: it re-hashes the local artifact and, when required, re-runs the bounded witness and compares its result digest. Marketplace metadata is therefore discovery input, never proof.

Offline proof:

```sh
go run ./cmd/marketdemo -selfcheck
go run ./cmd/marketdemo -json
```

Refusals cover duplicate identities, unknown seams/trust/error modes, incompatible ABI ranges, malformed digests, missing required witnesses, artifact mismatch, and witness-result mismatch. Executable in-process artifacts remain `trusted-compiled`; this descriptor does not turn them into a sandbox.
