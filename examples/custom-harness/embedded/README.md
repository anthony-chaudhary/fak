# Embedded custom harness: a behind-the-scenes agent

This example puts a fak agent inside an ordinary support workflow. The user sees a normal prepared reply; the host application does not expose a chat window, provider, model, or harness runtime.

```text
Ticket -> prepareReply -> replyAssistant -> harnessAgent -> semantic harness events
                  |                                      |
                  +---------- ordinary Reply <-----------+
```

## Run the learning example

From the repository root:

```bash
go run ./examples/custom-harness/embedded
go test ./examples/custom-harness/embedded
```

Both commands are offline and deterministic. They need no provider key, network, model, or GPU.

## What to notice

- `main.go` is ordinary host application code. Its only agent dependency is the small `replyAssistant` interface.
- `agent.go` is the narrow adapter. It consumes the public `pkg/harnesskit` semantic event contract and returns one application value.
- `offline.go` stands in for a generated harness runtime so the example stays reproducible.
- `main_test.go` is the behavior witness: the host invokes the agent adapter and produces the exact expected application result.

The host never imports internal fak packages or provider wire types. Unknown semantic events can be added without changing the host workflow; the adapter selects only the message and completion events it needs.

## Customize it

1. **Use your domain:** rename `Ticket`, `Reply`, and `prepareReply` in `main.go` to match a document workflow, background job, IDE action, desktop app, or other host flow.
2. **Change the agent contract:** keep the host-facing interface small. Prefer a domain result such as `Suggestion`, `Classification`, or `Summary` over exposing event or provider types.
3. **Connect a real harness:** generate a base product with `fak harness init --dir <your-dir> --module <your-module> --fak-version <version>`, then replace `offlineRunner.Run` with a transport adapter that returns the generated harness's ordered `harnesskit.Envelope` events.
4. **Choose the semantic result:** adjust the event switch in `harnessAgent.Suggest`. Keep validation, ordering checks, cancellation, and the completed-run requirement.
5. **Keep an offline witness:** use a deterministic runner in tests even after the application gains a live provider adapter.

The generated harness owns runtime and transport details. This example owns only the application-specific composition seam, which makes it a small parent pattern for many future embedded examples.
