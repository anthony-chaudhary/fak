# Captured run — `examples/custom-harness`

Real runs of the three presentation patterns (color stripped). Every pattern is
offline: no API key, no network, no model, no GPU. The `model.response` events are
the demo's own deterministic offline turn, not model output.

## 1. Terminal/TUI — `go run ./examples/custom-harness/tui`

Interactive in a real terminal; here driven from a pipe (a prompt, then `/quit`):

```console
$ printf 'What is a harness?\n/quit\n' | go run ./examples/custom-harness/tui
fak custom harness — terminal example
Type a prompt, or /quit to exit.
>   turn.started    What is a harness?
  model.response  example/custom-harness-tui received "What is a harness?"
  tool.requested  record_learning_example
  tool.completed  record_learning_example:ok
  turn.completed  ok
> bye
```

The five semantic events per turn are the harnesskit contract surface a real fak
host can inspect; `> bye` is the `/quit` path exiting 0.

## 2. Web UI — `go run ./examples/custom-harness/web`

The server blocks serving `127.0.0.1:8080`; a real turn via the rendered form:

```console
$ go run ./examples/custom-harness/web
web harness listening on http://127.0.0.1:8080

# from another terminal — the same turn the form submits:
$ curl -s -X POST http://127.0.0.1:8080/turn --data 'prompt=What is a harness?'
```

The response is the page re-rendered with the governed event stream filled in
(the interactive path a browser shows):

```html
<section aria-labelledby="events-heading">
  <h2 id="events-heading">Governed event stream</h2>
  <ol>
    <li><code>turn.started</code><span>What is a harness?</span></li><li><code>model.response</code><span>example/custom-harness-web received &#34;What is a harness?&#34;</span></li><li><code>tool.requested</code><span>record_learning_example</span></li><li><code>tool.completed</code><span>record_learning_example:ok</span></li><li><code>turn.completed</code><span>ok</span></li>
  </ol>
</section>
```

## 3. Embedded agent — `go run ./examples/custom-harness/embedded`

No UI at all — the composition pattern (an agent embedded in another program):

```console
$ go run ./examples/custom-harness/embedded
Re: changing a delivery address

Hello Sam,

You can update the address before the package ships.

— Support
```

## 4. The tests pin all of it

```console
$ go test ./examples/custom-harness/...
ok  	github.com/anthony-chaudhary/fak/examples/custom-harness/embedded	0.658s
ok  	github.com/anthony-chaudhary/fak/examples/custom-harness/event-consumer	0.239s
ok  	github.com/anthony-chaudhary/fak/examples/custom-harness/queue-worker	0.883s
ok  	github.com/anthony-chaudhary/fak/examples/custom-harness/replay-fixture	1.121s
ok  	github.com/anthony-chaudhary/fak/examples/custom-harness/tui	0.449s
ok  	github.com/anthony-chaudhary/fak/examples/custom-harness/web	1.213s
```

## What the capture proves

- **One product seam, three presentations.** TUI, web, and embedded each print the
  same governed event stream (`turn.started` → … → `turn.completed`) built on the
  same `pkg/harnesskit` product description — only the transport differs.
- **Fully offline and deterministic.** The offline turn is a fixed function of the
  prompt; the same input renders the same events every run, with no key, network,
  model, or GPU.
- **The web pattern is a server, not a one-shot.** `go run ./examples/custom-harness/web`
  blocks on `127.0.0.1:8080` until interrupted; that is its design, not a hang.
