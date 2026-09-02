# Custom harness UI/UX learning examples

These small, offline examples show three common ways to present a product built
on the public `pkg/harnesskit` seam:

| Parent pattern | Start here | Issue |
|---|---|---|
| Terminal/TUI | [`tui/`](tui/) | [#10561](https://github.com/anthony-chaudhary/fak/issues/10561) |
| Web UI | [`web/`](web/) | [#10562](https://github.com/anthony-chaudhary/fak/issues/10562) |
| Embedded agent | [`embedded/`](embedded/) | [#10563](https://github.com/anthony-chaudhary/fak/issues/10563) |

All three run deterministically without an API key, network, model, or GPU.
Generate the shared, compatibility-pinned base for your own product first:

```sh
fak harness init --dir ./my-harness --module example.com/my-harness
```

Then copy the presentation or composition pattern you need:

```sh
go run ./examples/custom-harness/tui
go run ./examples/custom-harness/web
go run ./examples/custom-harness/embedded
go test ./examples/custom-harness/...
```

## What you'll see

A captured run of all three patterns is in
[`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md): the same five governed events
(`turn.started` → … → `turn.completed`) rendered by a piped TUI turn, a live
`POST /turn` against the web server, and the embedded no-UI pattern — plus the
green `go test ./examples/custom-harness/...` lines that pin all of it. Each
pattern completes in under a second on a warm build cache; the first `go run`
of a pattern pays a one-time compile. The web pattern blocks serving
`127.0.0.1:8080` until interrupted — that is its design, not a hang.

## Scope — what this does not claim

These are UI/composition *starting points* over the public `pkg/harnesskit`
seam. What this does not claim: the `model.response` event is the demo's own
deterministic offline turn — no model runs, and nothing here demonstrates
streaming, approvals, authentication, persistence, full-screen terminal layouts,
provider integration, or any production harness behavior. It is also not a
performance or quality benchmark of any kind.

These directories are intentionally minimal parent categories. Streaming,
approvals, authentication, persistence, full-screen terminal layouts, and
application-specific integrations should become separate child examples rather
than making these starting points harder to understand or customize.
