---
title: "fak node — set up and connect to a fak serve node"
description: "One-command lifecycle for an always-on fak serve gateway: install once on the host, save it as the default router on each client, and connect from a home network or Tailscale-routed fleet."
---

# `fak node` — set up and connect to a node

> **Audience.** Operators installing an always-on `fak serve` gateway and connecting clients to it. By the end you can install, point a client at, run against, check, and tear down a node — from a single home box to a Tailscale-routed fleet.

`fak node` is the durable, one-command lifecycle for an always-on `fak serve` gateway. One
host runs the router for all paired clients; each client saves its destination once. It
replaces the per-platform shell scripts (`tools/install-mac-node.sh` and friends) with a
single Go verb that installs the gateway as a real system service, points a client at a
node, and tears it down — the same five commands whether the node is the laptop in front of
you or one box in a hyperscaler fleet.

| Command | What it does |
|---|---|
| `fak node install [--remote]` | Install the gateway as a system service on **this** host (macOS launchd, Linux systemd `--user`, Windows Scheduled Task). `--remote` binds `0.0.0.0`, generates a bearer key, and prints client connection lines. |
| `fak node use HOST[:PORT] [--key KEY]` | On a **client**, save its default router in `~/.config/fak/node.json` and print the export lines. Probes `GET /healthz` and warns if the node is unreachable. |
| `fak node run -- CMD [ARGS…]` | Launch `CMD` (e.g. `claude`) with `ANTHROPIC_BASE_URL` (and `ANTHROPIC_API_KEY`, when a key is set) pointed at the configured node. Exits with the child's status. |
| `fak node status` | Service state (launchd/systemd/schtasks) + `/healthz` for loopback and the configured node. |
| `fak node forget` | Clear `~/.config/fak/node.json`. |

By default, the installed gateway is `fak serve --provider anthropic`: an adjudication
proxy in front of `api.anthropic.com`, with the bundled capability policy applied to every
tool call. On that Anthropic wire the gateway forwards the caller's upstream credential;
the generated gateway bearer is not an Anthropic API key. The shared local-model path
below instead uses a model backend on the host and authenticates clients with the gateway
bearer.

## At home — one box, no network

The smallest useful setup: run the gateway and a guarded agent on the same machine.

```bash
export ANTHROPIC_API_KEY="sk-ant-..."   # local caller's upstream credential
fak node install                        # loopback gateway on 127.0.0.1:8080
fak node status                         # service up + /healthz 200

fak manage claude                       # guarded interactive session
```

`install` with no flags binds loopback only — nothing is exposed off-host, and no bearer key
is needed. `fak manage` wraps the agent so the kernel adjudicates every tool call locally.

## At home — one router for other devices (Tailscale)

Run one gateway on an always-on box such as Halo, a Mac mini, or a desktop. A laptop or
second desktop connects to that gateway over a protected network. The example assumes an
OpenAI-compatible model backend is already listening on the host's loopback address; choose
any model and device supported by that backend.

On **Halo or the chosen host**:

```bash
fak node install --remote \
  --base-url http://127.0.0.1:8131/v1 --model your-model-id
# one gateway service on 0.0.0.0:8080; prints a generated bearer key and client lines
```

Alternatively, start the router directly on the host (without installing a service):

```bash
export FAK_GATEWAY_KEY="$(openssl rand -hex 32)"
fak serve --addr 0.0.0.0:8080 --provider openai \
  --base-url http://127.0.0.1:8131/v1 --model your-model-id \
  --require-key-env FAK_GATEWAY_KEY
```

On each **client** (laptop, other desktop), pair once using the host's protected network
address and the bearer key. Replace the example address and key with the values printed
by the host's installer:

```bash
fak node use 100.64.0.10:8080 --key 'PASTE_PRINTED_GATEWAY_KEY'
fak agent                               # saved router is the default
fak chat                                # same saved router
```

`use` writes the router to `~/.config/fak/node.json`. Bare `fak agent` and `fak chat` use
that saved router; neither client needs its own `fak serve` process. An explicit
`--base-url` or provider base URL environment variable overrides the saved default for a
run. `fak node run -- claude` still reads the same config for an external client, and
`fak node forget` clears the saved default.

Keep the bearer on a protected path such as a tailnet, VPN, or TLS reverse proxy; HTTP
on an unprotected LAN carries it in cleartext. `--remote` listens on every interface, so
restrict the gateway listener to trusted clients with the host firewall.
Do not enable `--allow-lan`, which exempts local-network callers from bearer
authentication. A direct LAN bind of `fak up` does not provide this gateway bearer
check; use the authenticated `fak node install --remote` or `fak serve` path above for
off-host clients.

## Disaggregated / hyperscaler — a fleet of nodes

The same primitives scale to a fleet. Each node is an independent always-on gateway with its
own bearer key, reachable over the tailnet (or any routable network); clients pick a node by
pointing `use` at it.

Per node (one `install --remote` each, with a local model backend on each host):

```bash
# on node-a, node-b, … (Linux)
fak node install --remote --port 8080 \
  --base-url http://127.0.0.1:8131/v1 --model your-model-id
systemctl --user status fak-serve-gateway      # or: fak node status
```

From a client or a dispatcher, target whichever node should serve a given session:

```bash
fak node use node-a.tailnet:8080 --key "$NODE_A_KEY"
fak node run -- claude
# … later, move the session to a different node:
fak node use node-b.tailnet:8080 --key "$NODE_B_KEY"
fak node run -- claude
```

Each node serves its own model backend, and clients carry a per-node gateway bearer.
Adding or rotating a node changes the client's selected URL and bearer through `use`.
For HA, route clients across independently managed nodes with `use`. See
[deployment-guide.md](deployment-guide.md) and [advanced-topics.md](advanced-topics.md) for
multi-region and HA patterns, and [security.md](security.md) for the network threat model.

## Where things live

| Path | Written by | Purpose |
|---|---|---|
| `~/.config/fak/node.json` (`%APPDATA%\fak\node.json` on Windows) | `fak node use` | the client's default router `{url, key}` — read by `agent`, `chat`, `run`, and `status` |
| `~/.config/fak/node-policy.json` | `fak node install` | the capability policy the gateway enforces |
| `~/.config/fak/logs/serve.log` · `serve.err` · `serve_audit.jsonl` | the gateway | stdout/stderr and the kernel decision journal |
| launchd `com.fak.serve-gateway` · systemd `fak-serve-gateway` · schtasks `FakServeGateway` | `fak node install` | the always-on service definition |

## Uninstall

```bash
fak node install --uninstall    # removes the service on this host
fak node forget                 # clears the client's node.json
```

## See also

- [server-quickstart.md](server-quickstart.md) — the fastest path to a running gateway
- [server-config.md](server-config.md) — every `fak serve` flag and env var
- [policy-guide.md](policy-guide.md) — author the capability policy the node enforces
- [always-on-dogfood-server.md](always-on-dogfood-server.md) — the always-on gateway + guarded fleet design
- [deployment-guide.md](deployment-guide.md) — production deployment
