# fakclient — Go SDK for the fak gateway

[![Go Reference](https://pkg.go.dev/badge/github.com/anthony-chaudhary/fak/pkg/fakclient.svg)](https://pkg.go.dev/github.com/anthony-chaudhary/fak/pkg/fakclient)

A small, stdlib-only Go client for the **fak-native verdict surface** of `fak serve`
— the `/v1/fak/*` endpoints that adjudicate a tool call and return a verdict as a
value. Part of the SDK story for [F-007](https://github.com/anthony-chaudhary/fak/issues/205).

```bash
go get github.com/anthony-chaudhary/fak/pkg/fakclient
```

```go
c := fakclient.New("http://127.0.0.1:8080")

// Pre-execution gate: run your own tool only if fak allows it.
resp, err := c.Adjudicate(ctx, fakclient.SyscallRequest{
    Tool:      "write_file",
    Arguments: json.RawMessage(`{"path":"report.csv"}`),
})
if err != nil {            // a real fault (400/401/5xx), never a refusal
    log.Fatal(err)
}
if !resp.Verdict.Allowed() {
    // A refusal is a successful 200 carried in the verdict — handle it as a value.
    log.Printf("refused: %s (%s)", resp.Verdict.Kind, resp.Verdict.Reason)
    return
}
```

When the gateway is booted with `--require-key-env`, pass the bearer token and an
optional tenant principal:

```go
c := fakclient.New(url,
    fakclient.WithAPIKey(os.Getenv("FAK_KEY")),
    fakclient.WithPrincipal("tenant-7"))
```

## What it covers

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `Adjudicate` | `POST /v1/fak/adjudicate` | Pre-execution verdict only |
| `Syscall` | `POST /v1/fak/syscall` | Adjudicate **and** execute |
| `Admit` | `POST /v1/fak/admit` | Contain a client-executed result |
| `Changes` | `GET /v1/fak/changes` | Drain the cross-agent change feed |
| `Revoke` | `POST /v1/fak/revoke` | Refute a world-state witness |
| `Models` / `Health` | `GET /v1/models`, `/healthz` | List the model, liveness |

The OpenAI / Anthropic / Gemini surfaces fak fronts are **not** wrapped here on
purpose: a caller who already uses an official OpenAI, Anthropic, or google-genai
SDK adopts fak by repointing that SDK's base URL at `fak serve` (see
[`docs/integrations`](../../docs/integrations)). This package is only the
fak-native surface, which has no off-the-shelf client.

## Source of truth & drift gate

Every request/response type mirrors the gateway's wire DTOs
(`internal/gateway/wire.go`), documented in the OpenAPI spec
[`docs/fak/openapi.yaml`](../../docs/fak/openapi.yaml). The agreement is gated:
`wireparity_test.go` round-trips the server's own types through these and fails
the build on any drift — the in-code companion to the route-drift gate in
`internal/gateway/openapi_spec_test.go`.

## The Python and TypeScript SDKs

The same OpenAPI spec is the generator input for the other two languages:

```bash
# Python
openapi-python-client generate --path docs/fak/openapi.yaml

# TypeScript
npx @openapitools/openapi-generator-cli generate \
    -i docs/fak/openapi.yaml -g typescript-fetch -o sdk/ts
```

Publishing the generated Python/TypeScript packages to PyPI / npm (and tagging
this Go module for pkg.go.dev) is a release-pipeline step that needs registry
credentials, so it is not done in-tree.
