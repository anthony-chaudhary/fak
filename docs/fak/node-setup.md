---
title: "fak node — set up and connect to a fak serve node"
description: "One-command lifecycle for an always-on fak serve gateway: install on the host, use to point a client at it, run to launch the client, status to check, forget to disconnect — from a single home box to a Tailscale-routed fleet."
---

# `fak node` — set up and connect to a node

> **Audience.** Operators installing an always-on `fak serve` gateway and connecting clients to it. By the end you can install, point a client at, run against, check, and tear down a node — from a single home box to a Tailscale-routed fleet.

`fak node` is the durable, one-command lifecycle for an always-on `fak serve` gateway. It
replaces the per-platform shell scripts (`tools/install-mac-node.sh` and friends) with a
single Go verb that installs the gateway as a real system service, points a client at a
node, and tears it down — the same five commands whether the node is the laptop in front of
you or one box in a hyperscaler fleet.

| Command | What it does |
|---|---|
| `fak node install [--remote]` | Install the gateway as a system service on **this** host (macOS launchd, Linux systemd `--user`, Windows Scheduled Task). `--remote` binds `0.0.0.0`, generates a bearer key, and prints client connection lines. |
| `fak node use HOST[:PORT] [--key KEY]` | On a **client**, record the node in `~/.config/fak/node.json` and print the export lines. Probes `GET /healthz` and warns if the node is unreachable. |
| `fak node run -- CMD [ARGS…]` | Launch `CMD` (e.g. `claude`) with `ANTHROPIC_BASE_URL` (and `ANTHROPIC_API_KEY`, when a key is set) pointed at the configured node. Exits with the child's status. |
| `fak node status` | Service state (launchd/systemd/schtasks) + `/healthz` for loopback and the configured node. |
| `fak node forget` | Clear `~/.config/fak/node.json`. |

The gateway it installs is `fak serve --provider anthropic`: a local adjudication proxy in
front of `api.anthropic.com`, with the bundled capability policy applied to every tool call.
The upstream credential (`ANTHROPIC_API_KEY`, or a Claude subscription token) lives on the
**host**; clients present only the gateway's bearer key, never the upstream secret.

## At home — one box, no network

The smallest useful setup: run the gateway and a guarded agent on the same machine.

```bash
export ANTHROPIC_API_KEY="sk-ant-..."   # the host's upstream credential
fak node install                        # loopback gateway on 127.0.0.1:8080
fak node status                         # service up + /healthz 200

fak guard -- claude                     # guarded interactive session
```

`install` with no flags binds loopback only — nothing is exposed off-host, and no bearer key
is needed. `fak guard` wraps the agent so the kernel adjudicates every tool call locally.

## At home — a host plus other devices (Tailscale)

Run the gateway on one always-on box (a Mac mini, a home server) and connect from a laptop,
a phone client, or a second desktop over your tailnet.

On the **host**:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
fak node install --remote               # binds 0.0.0.0:8080, generates FAK_GATEWAY_KEY
# prints the Tailscale-routable client lines + the bearer key — copy them
```

On each **client** (laptop, other desktop):

```bash
fak node use <host-tailscale-ip>:8080 --key <FAK_GATEWAY_KEY>
fak node run -- claude
```

`use` writes the node to `~/.config/fak/node.json` and prints the same export lines for
shells or CI that prefer them. `run` reads that config and launches the client against the
node with zero environment juggling — the config `use` writes is what `run` consumes.

## Disaggregated / hyperscaler — a fleet of nodes

The same primitives scale to a fleet. Each node is an independent always-on gateway with its
own bearer key, reachable over the tailnet (or any routable network); clients pick a node by
pointing `use` at it.

Per node (one `install --remote` each — systemd on Linux hosts, launchd on Mac verify nodes):

```bash
# on node-a, node-b, … (Linux)
export ANTHROPIC_API_KEY="sk-ant-..."
fak node install --remote --port 8080
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

Because the upstream credential stays on each host and clients carry only a per-node bearer,
adding or rotating a node never touches the clients beyond a one-line `use`. For HA, multiple
nodes can share one upstream account; route clients across them with `use`. See
[deployment-guide.md](deployment-guide.md) and [advanced-topics.md](advanced-topics.md) for
multi-region and HA patterns, and [security.md](security.md) for the network threat model.

## Where things live

| Path | Written by | Purpose |
|---|---|---|
| `~/.config/fak/node.json` (`%APPDATA%\fak\node.json` on Windows) | `fak node use` | the client's configured node `{url, key}` — read by `run` and `status` |
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
