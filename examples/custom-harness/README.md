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

These directories are intentionally minimal parent categories. Streaming,
approvals, authentication, persistence, full-screen terminal layouts, and
application-specific integrations should become separate child examples rather
than making these starting points harder to understand or customize.
