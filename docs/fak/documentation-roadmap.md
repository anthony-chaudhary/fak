---
title: "fak documentation roadmap and shipped doc index"
description: "Tracks fak server and agent-integration documentation: which guides have shipped, the historical structure vision, and remaining work items."
---

# fak Documentation Roadmap

**Status:** Complete — all tracked documentation has shipped and is wired into the index

**Last updated:** 2026-06-21

---

## Overview

This roadmap tracks GitHub issues for remaining documentation work to achieve the "100x better documentation" vision for both objectives:

1. **fak server documentation** — Gateway deployment, configuration, and operation
2. **Agent integration documentation** — Integrating fak with coding agents and frameworks

---

## Completed Documentation ✅

The following documentation already exists in `docs/fak/` and `docs/`:

| Document | Location | Purpose |
|----------|----------|---------|
| **Getting Started Tutorial** (#161) | `docs/fak/tutorial.md` | **Guided zero-to-first-call session, real captured output at every step** ✅ |
| **Policy Authoring Guide** (#162) | `docs/fak/policy-guide.md` | **Worked policy examples with real `--check`/`preflight` output** ✅ |
| **Observability Guide** (#163) | `docs/fak/observability.md` | **Metrics / logs / traces with real `/metrics` + `/debug/vars` output** ✅ |
| **API Reference** (#164) | `docs/fak/api-reference.md` | **Every gateway endpoint, request, and response** ✅ |
| **Multi-Language Examples** (#165) | `docs/fak/multi-language-examples.md` | **Runnable client code in Python, JavaScript, Go, and Rust** ✅ |
| **Agent Framework Integration** (#166) | `docs/fak/agent-framework-integration.md` | **Per-framework cookbook: LangChain/LangGraph, LlamaIndex, AutoGen, CrewAI, Semantic Kernel, Haystack, Griptape** ✅ |
| **Advanced Topics** (#167) | `docs/fak/advanced-topics.md` | **Performance tuning, scaling, multi-region, and HA** ✅ |
| **Security Best Practices** (#168) | `docs/fak/security.md` | **Threat model, auth, hardening checklist with real verdict output** ✅ |
| **FAQ and Common Issues** (#169) | `docs/fak/faq.md` | **Short, honest answers to the most-asked questions** ✅ |
| Deployment Guide | `docs/fak/deployment-guide.md` | Production deployment for `fak serve` ✅ |
| Migration Guide | `docs/fak/migration-guide.md` | Migrating an existing stack (OpenAI API, LangChain, AutoGen, llama.cpp) onto `fak` ✅ |
| Docs Index | `docs/fak/README.md` | Navigation hub for the operator/integrator docs ✅ |
| Server Quickstart | `docs/fak/server-quickstart.md` | Fast path to running fak serve |
| Server Troubleshooting | `docs/fak/server-troubleshooting.md` | Debugging common issues |
| Serve Configuration | `docs/serve-config.md` | Env vars, auth, timeouts, reload |
| MCP Tool Result | `docs/mcp-tool-result.md` | MCP wire format reference |
| Claude Integration | `docs/integrations/claude.md` | Claude Code + Anthropic API setup |
| Policy Reference | `fak/POLICY.md` | Policy schema and refusal vocabulary |
| Architecture Overview | `fak/ARCHITECTURE.md` | System design and extension model |
| SOTA Optimizations | `docs/explainers/sota-optimizations.md` | What "tuned" means |

---

## Remaining Documentation Issues

**None open.** Every tracked page has shipped — see *Completed Documentation* above. The
final pass landed the server-documentation set (#164 API reference, #167 advanced topics,
#169 FAQ) and the agent-integration set (#165 multi-language examples, #166 framework
guides). All are wired into [`README.md`](README.md).

> **Shape note:** #165 and #166 shipped as single consolidated pages
> ([`multi-language-examples.md`](multi-language-examples.md) and
> [`agent-framework-integration.md`](agent-framework-integration.md)) rather than the
> per-language / per-framework file tree the original *Documentation Structure Vision*
> below sketched. The consolidated pages cover the same surface; the vision tree is kept
> below as a historical record, not a live to-do list.

---

## Issue Summary by Category

### Getting Started
- **#161** - Tutorial from zero to first agent
- **#169** - FAQ for common questions

### Core Reference
- **#164** - Complete API reference with OpenAPI spec

### Configuration & Operations
- **#162** - Policy authoring patterns
- **#163** - Monitoring, metrics, debugging
- **#167** - Scaling and performance
- **#168** - Security best practices

### Integration
- **#165** - Python, JS, Go, Rust examples
- **#166** - LangChain, LlamaIndex, AutoGen, CrewAI

---

## Progress Tracking

**Total issues:** 9
**Completed:** 9  (#161 tutorial, #162 policy guide, #163 observability, #164 API reference, #165 multi-language examples, #166 framework guides, #167 advanced topics, #168 security, #169 FAQ)
**In progress:** 0
**Remaining:** 0

### Priority Breakdown (remaining)
- None — the documentation backlog is drained.

---

## Documentation Structure Vision

When complete, the `docs/fak/` directory should contain:

```
docs/fak/
├── README.md                        # Index and navigation
├── tutorial.md                      # #161 - Getting started tutorial
├── server-quickstart.md             # ✅ Existing
├── server-troubleshooting.md        # ✅ Existing
├── policy-guide.md                  # #162 - Policy authoring
├── observability.md                 # #163 - Monitoring and metrics
├── openapi.yaml                     # #164 - API spec
├── advanced-topics.md               # #167 - Scaling and performance
├── security.md                      # #168 - Security best practices
├── faq.md                           # #169 - FAQ
├── examples/
│   ├── python/                      # #165 - Python examples
│   ├── javascript/                  # #165 - JS examples
│   ├── go/                          # #165 - Go examples
│   └── rust/                        # #165 - Rust examples
└── integrations/
    ├── claude.md                    # ✅ Existing
    ├── langchain.md                 # #166 - LangChain guide
    ├── llamaindex.md               # #166 - LlamaIndex guide
    ├── autogen.md                  # #166 - AutoGen guide
    └── crewai.md                   # #166 - CrewAI guide
```

---

## Next Steps

The original documentation backlog is closed. Ongoing maintenance, not net-new pages:

1. **Keep captured output current** — re-capture command/output blocks on each release
   (the tutorial/policy-guide/observability/security pages pin a `fak` version).
2. **Cross-link as new pages land** — wire any future `docs/fak/` page into
   [`README.md`](README.md) so the index never drifts from disk (this pass closed that gap).
3. **Promote, don't duplicate** — new framework or language coverage extends the existing
   consolidated pages rather than re-introducing a per-file tree.

---

## Contributing

When working on these issues:

1. Check the issue for the full specification and success criteria
2. Create documentation in the specified location
3. Update cross-references between documents
4. Link the PR to the issue
5. Update this roadmap's progress tracking

---

*This roadmap is auto-generated from the documentation goal worker. For questions or updates, see the linked GitHub issues.*
