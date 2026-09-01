# Minimal terminal custom harness

A line-oriented TUI learning example built on the public `pkg/harnesskit`
contract. It is deliberately just standard input and output: no terminal
framework, API key, network, model, or GPU is required.

## Run it

From the repository root:

```sh
go run ./examples/custom-harness/tui
```

Enter prompts and read the governed semantic lifecycle. Type `/quit` to exit.

## Test the rendered transcript

```sh
go test ./examples/custom-harness/tui
```

The test scripts a complete session and captures the bytes a learner sees. That
render witness proves the prompt, semantic events, and clean exit stay visible.

## Customize it

1. **Name:** edit the heading in `run` and the product ID in `newApp`.
2. **Rendering:** change the single `fmt.Fprintf` line that renders events.
3. **Commands:** add small slash commands next to `/quit`.
4. **Behavior:** replace only `runOfflineTurn` with your host/provider adapter;
   keep presentation separate and keep tool authority behind fak.

To regenerate the compatibility-pinned custom-harness base first:

```sh
fak harness init --dir ./my-harness --module example.com/my-harness
```

This directory is the simplest parent for future terminal children such as a
full-screen TUI, streaming output, approval prompts, tool panes, and multi-agent
views. Those features belong in separate examples so this one remains copyable.
