# Contextual harness preview

`fak harness preview` is the decision seam between a resolved contextual product
lock and launch. It keeps the common path quiet: when the verified candidate
lock has the same canonical ID as the last admitted lock, the command writes
zero bytes and exits 0.

A decision is requested only when the comparison finds one of four reasons:

- `novel-domain`: the contextual domain changes (for example coding to legal);
- `conflict`: resolution could not produce a compatible effective product;
- `privilege-widening`: tool or credential access appears, policy grants grow,
  a denial or locked floor disappears, or a mandatory workflow is weakened;
- `low-confidence`: classification requests a choice or confidence is below
  the 0.75 automatic-selection floor.

Each change names the source layer, changed capability, operational
consequence, and the reversible `keep-current` choice. The other bounded
choices are `approve-once` and an expiring scope-local `remember`; this command
does not maintain global learned state or provide a settings dashboard.

## Launch gate

Resolve a candidate, compare it with the admitted lock, and launch only after
exit 0 or an explicit operator decision:

```sh
fak harness preview \
  --current admitted.lock.json \
  --candidate candidate.lock.json \
  --current-domain coding \
  --candidate-domain legal
```

An unchanged candidate emits nothing. A decision exits 3, so a shell pipeline
cannot accidentally continue to launch. Interactive output is deliberately one
bounded block:

```text
contextual harness decision required
- novel-domain | domain:legal | domain:legal
  switch contextual defaults from coding to legal; choice: keep the current lock
choices: approve-once | remember | keep-current
```

Use `--view tui` for the ANSI-free pane projection. It has the same information
and emits exactly one block. Use `--headless` (or `--view json`) for the
machine-readable `fak.harness-preview/v1alpha1` result. A required decision
still exits 3 and includes recovery action IDs, so automation fails closed
rather than hiding a prompt.

A resolver conflict can be surfaced before a candidate lock exists:

```sh
fak harness preview --conflict "route requires incompatible contract v2" --headless
```

Classification JSON from `fak harness classify` can be supplied with
`--classification`; ambiguous or low-confidence classification is then part of
the same decision instead of becoming a second prompt.

## Security and reversal

Both lock files are schema-checked and their canonical SHA-256 IDs are verified
before equality is trusted. Preview never changes either lock. `keep-current`
is therefore a no-write reversal. Implementations that persist `remember`
should store an expiring, scope-local admission receipt; deleting that receipt
restores the prior behavior. An admission must never relax a company policy
floor—the typed composition and lock-resolution stages remain authoritative.

## Captured witnesses

`internal/harnesspreview/harnesspreview_test.go` covers all four reasons,
privilege widening, zero-byte unchanged CLI/TUI renders, and an exact bounded
TUI render. `cmd/fak/harness_preview_test.go` captures the command surface and
proves unchanged output is empty, risky TUI output has no escape bytes, and
headless decisions exit 3 with recovery actions.
