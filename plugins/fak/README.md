# fak — Claude Code plugin

Put the [Fused Agent Kernel](https://github.com/anthony-chaudhary/fak) in front of
Claude Code's tool calls. This plugin registers fak's MCP server (`fak serve
--stdio`), which exposes the kernel's adjudication verbs as MCP tools:

- `fak_adjudicate` — screen a proposed tool call against the capability policy
  *before* the agent runs it (default-deny).
- `fak_syscall` — run a tool call *through* the kernel so the verdict and the
  result are governed on one path.
- `fak_admit` — screen a result the agent executed itself, holding a suspicious
  tool result out of context (prompt-injection / tool-poisoning quarantine).

The kernel decides which effects are allowed and which results may enter the
model's context by structure, not by a classifier the model can argue past.

## Prerequisite: `fak` on your PATH

This plugin does **not** bundle a binary — it runs whatever `fak` your shell
resolves, so the same artifact you run elsewhere is what the plugin governs.
Install it first:

```bash
# Go toolchain (the module root is the repo root):
go install github.com/anthony-chaudhary/fak/cmd/fak@latest

# or download a release binary for your platform and put it on PATH:
#   https://github.com/anthony-chaudhary/fak/releases
```

Verify: `fak --version`.

## Install the plugin

```text
/plugin marketplace add anthony-chaudhary/fak
/plugin install fak@fak
```

Then restart Claude Code. The `fak` MCP server starts on demand over stdio (no
network, no API key, no listener). To confirm it loaded, run `/mcp` and look for
the `fak` server and its `fak_*` tools.

## Add a policy (recommended)

By default fak's MCP server runs with no policy file, which is fine for trying the
verbs out. For real governance, point it at an allow-list. Edit the installed
plugin's `plugin.json` `mcpServers.fak.args`, or wire fak directly in your
project's `.mcp.json` instead (see
[examples/mcp](https://github.com/anthony-chaudhary/fak/tree/main/examples/mcp)):

```json
"args": ["serve", "--stdio", "--policy", "/abs/path/to/your-policy.json"]
```

Author and review a policy with `fak policy` — see
[POLICY.md](https://github.com/anthony-chaudhary/fak/blob/main/POLICY.md).

## Learn more

- [What fak is and the two flips](https://github.com/anthony-chaudhary/fak#readme)
- [MCP tool-result wire](https://github.com/anthony-chaudhary/fak/blob/main/docs/mcp-tool-result.md)
- [Integration index](https://github.com/anthony-chaudhary/fak/blob/main/docs/integrations/README.md)
