# Minimal web custom harness

A framework-free learning example that connects an HTML form to the public
`pkg/harnesskit` contract. The turn is intentionally offline and deterministic:
it needs no API key, external network, JavaScript toolchain, or GPU.

## Run it

From the repository root:

```sh
go run ./examples/custom-harness/web
```

Open `http://127.0.0.1:8080`, enter a prompt, and submit it. The page shows the
semantic events for one governed turn:

`turn.started` → `model.response` → `tool.requested` → `tool.completed` → `turn.completed`

## Test the HTTP boundary

```sh
go test ./examples/custom-harness/web
```

The test uses `httptest`; it posts a prompt and checks the rendered event stream.
No live server or browser is required.

## Customize it

Start with one small change at a time:

1. **Name and product identity:** edit the title/text in `index.html` and the
   product ID/profile in `newApp`.
2. **Look and feel:** edit the embedded CSS in `index.html`. There is no asset
   pipeline to configure.
3. **Offline behavior:** edit `runOfflineTurn` while learning. Keep its event
   names stable so the UI continues to explain the governed lifecycle.
4. **Real behavior:** replace only `runOfflineTurn` with your host/provider
   adapter. Keep the `net/http` handlers as the presentation boundary and keep
   authority behind fak rather than calling tools directly from a handler.

To regenerate a fresh base custom harness before applying this web shell:

```sh
fak harness init --dir ./my-harness --module example.com/my-harness
```

The generated product is the compatibility-pinned base. This example focuses on
how a browser UI can present that product's semantic turn events. Later child
examples can add streaming, approvals, auth, state, or deployment without making
this first example harder to copy.
