---
title: "Publishing fak to the Official MCP Registry"
description: "How fak is listed in the Model Context Protocol registry (modelcontextprotocol/registry): the OCI image route, the server.json manifest, and the one interactive publish step a maintainer runs."
---

# Publishing fak to the Official MCP Registry

> **Audience.** A fak maintainer publishing the server to the Official MCP Registry. By the end you'll know what is already wired in the repo and the one interactive publish step a human has to run.

The [Official MCP Registry](https://github.com/modelcontextprotocol/registry) is the
canonical index MCP clients query to discover servers. fak is listed there as an **OCI
package** — the same container the [deployment guide](deployment-guide.md) describes,
published to GitHub Container Registry.

Most of this is already wired in the repo. The only step that can't be automated is the
final publish, because it needs a human to authenticate the `io.github.anthony-chaudhary/*`
namespace via GitHub device flow.

## What's already in the repo

| Piece | Where | What it does |
|---|---|---|
| OCI ownership label | [`Dockerfile`](https://github.com/anthony-chaudhary/fak/blob/main/Dockerfile) | `LABEL io.modelcontextprotocol.server.name="io.github.anthony-chaudhary/fak"` — proves the image belongs to the namespace. |
| Image publish | [`.github/workflows/release-container.yml`](https://github.com/anthony-chaudhary/fak/blob/main/.github/workflows/release-container.yml) | On a `v*` tag, builds + pushes `ghcr.io/anthony-chaudhary/fak:{version,latest}`. |
| Registry manifest | [`server.json`](https://github.com/anthony-chaudhary/fak/blob/main/server.json) | The server metadata the registry stores: name, description, repo, and the `oci` package pointing at the ghcr image with a `stdio` transport running `fak serve --stdio`. |

## The one-time publish (maintainer step)

After a release has cut a `vX.Y.Z` tag and `release-container.yml` has pushed the image:

1. **Make the ghcr package public** (first publish only). In the repo's
   *Packages* tab, set the `fak` container package visibility to public so registry
   clients can pull it.

2. **Install the publisher CLI:**
   ```bash
   brew install mcp-publisher
   # or: curl -L "https://github.com/modelcontextprotocol/registry/releases/latest/download/mcp-publisher_$(uname -s | tr '[:upper:]' '[:lower:]')_$(uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/').tar.gz" | tar xz mcp-publisher && sudo mv mcp-publisher /usr/local/bin/
   ```

3. **Authenticate the namespace** (interactive GitHub device flow — this is the step that
   needs a human):
   ```bash
   mcp-publisher login github
   ```
   Authenticating as the `anthony-chaudhary` GitHub account authorizes the
   `io.github.anthony-chaudhary/*` namespace, which must match the `name` in `server.json`.

4. **Publish** from the repo root (it reads `server.json`):
   ```bash
   mcp-publisher publish
   ```

## Updating on each release

Bump the `version` and the `oci` `identifier` tag in `server.json` to the new `vX.Y.Z`
(matching what `release-container.yml` pushed), then re-run `mcp-publisher publish`. The
namespace login persists, so step 3 is a one-time cost.

## Why OCI and not a bare repo

The registry's `server.json` requires a real package artifact — one of `npm`, `pypi`,
`nuget`, `cargo`, `oci`, or `mcpb`. fak ships as a Go binary, so the natural fit is the
container image (`oci`); ownership is verified by the Dockerfile label rather than a
package-manager account. A GitHub repo alone is not a publishable package type.

## Related directories

- **Smithery** — see [`smithery.yaml`](https://github.com/anthony-chaudhary/fak/blob/main/smithery.yaml); lists fak as a stdio server.
- **mcp.so / mcpservers.org / Glama** — web-form directories; see the distribution notes.
