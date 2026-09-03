# Concept Study: Portkey AI Gateway

**Repository:** https://github.com/portkey-ai/gateway  
**Pinned Revision:** `669825cbe89ee51569918b8f78a9db486fd69dd4`  
**Study Date:** 2026-09-02  
**Status:** Studied  

---

## Repository Overview

Portkey AI Gateway is a **multi-provider AI gateway** built on **Hono** (Cloudflare Workers/Node.js) that routes requests to 80+ AI providers through a unified OpenAI-compatible API. It processes 10B+ tokens/day with <1ms latency overhead.

**Key Capabilities:**
- **Unified Provider Abstraction** - 1600+ models across 45+ providers via declarative config
- **Routing Strategies** - Fallback, Load Balance, Conditional, Single
- **Reliability** - Retries with exponential backoff, retry-after header support, circuit breaker
- **Guardrails** - 20+ third-party + built-in plugins, fail-open design, sequential/parallel execution
- **Caching** - Two-tier: simple in-memory + pluggable backends (memory/file/Redis/Cloudflare KV)
- **Observability** - OTLP spans with GenAI semantic conventions, real-time log streaming
- **MCP Gateway** - Centralized MCP server auth, access control, observability

---

## Source Classes Covered

| Class | Coverage | Notes |
|-------|----------|-------|
| readme_docs | ✅ | README.md, docs/ |
| architecture_design | ✅ | Core routing, provider abstraction, middleware chain |
| runtime_source | ✅ | src/ (766 files), all load-bearing subsystems |
| tests_fixtures | ⚠️ | src/tests/ exists but not exhaustively read |
| history_changelog_releases | ⚠️ | Git history not deep-cloned (--depth 1) |
| open_closed_issues_prs_discussions | ⚠️ | Not read; would need deep clone |
| roadmap_todos | ⚠️ | 2.0.0 pre-release branch noted |
| license_provenance | ✅ | MIT License (permissive) |
| fak_selfquery_witness | ✅ | fak capabilities/feature queries run for key axes |
| candidate_matrix | ✅ | 14 candidates extracted across 5 subsystems |
| completeness_critic | ✅ | See below |
| issue_tracking | ✅ | 9 issues filed (#10692-#10700) |

---

## Completeness Critic

**Subsystems Read (Fan-out):**
1. Core Routing Engine (handlers, middlewares, services) - ✅ Deep
2. Provider Abstraction Layer (45+ providers, transforms) - ✅ Deep
3. Caching (middleware + unified cache service) - ✅ Deep
4. Guardrails/Hooks (plugin architecture) - ✅ Deep
5. Observability/Logging (OTLP, real-time streaming) - ✅ Deep
6. MCP Gateway (types, adapters, cache) - ⚠️ Shallow (distributed)
7. Request/Response Transformation - ✅ Deep
8. Streaming Architecture (SSE + AWS EventStream) - ✅ Deep

**Subsystems Not Read (Justified):**
- **Tests** - Large test suite; not core to architectural borrow candidates
- **Cookbooks/Examples** - Usage documentation, not implementation
- **Deployment configs** (Docker, Wrangler, K8s) - Operational, not kernel-relevant
- **PR/History** - Shallow clone; would need --depth removal for "why they changed it"

**Verdict:** All load-bearing subsystems for gateway/kernel-relevant borrows were deeply read. The completeness critic finds **no material kernel-relevant subsystem unopened**.

---

## Candidate Matrix

| # | Borrow Technique | Source path:line@sha | Axis | Fak Status | Disposition | Worldview Reason |
|---|------------------|---------------------|------|------------|-------------|------------------|
| 1 | **Recursive target resolution with config inheritance** | `src/handlers/handlerUtils.ts:476-834@669825c` | Nested routing with inherited retry/cache/hooks | ABSENT | INSPIRE | They built for multi-target enterprise routing; fak optimizes single-model session efficiency |
| 2 | **Conditional routing DSL (MongoDB-like queries on metadata/params/url)** | `src/services/conditionalRouter.ts:1-156@669825c` | Rule-based routing on request context | ABSENT | INSPIRE | Their users need dynamic per-request routing; fak's users need deterministic replay |
| 3 | **Post-hook retry (retries include guardrail transformations)** | `src/handlers/handlerUtils.ts:1265@669825c` | Retry after hook mutations | PARTIAL | INSPIRE | They mutate on retry for guardrail compliance; fak retries raw provider errors |
| 4 | **Gateway exception header (`x-portkey-gateway-exception`) to stop fallback chain** | `src/handlers/handlerUtils.ts:678@669825c` | Unrecoverable error signaling | ABSENT | INSPIRE | Enterprise needs explicit unrecoverable signaling; fak uses error types |
| 5 | **Declarative provider parameter config with transforms/validation/defaults** | `src/providers/types.ts:38-160@669825c` | Normalization of 1600+ models | PARTIAL | INSPIRE | They normalize across 45 providers; fak owns the model runtime |
| 6 | **Dynamic provider config selection (Bedrock-style `getConfig`)** | `src/providers/bedrock/index.ts:98-272@669825c` | Single provider entry for multi-model families | ABSENT | INSPIRE | Bedrock hosts 6 model families; fak doesn't proxy multi-model providers |
| 7 | **Two-phase response transforms (non-stream + streaming chunk transformers)** | `src/providers/anthropic/chatComplete.ts:538-831@669825c` | Streaming normalization with state | PARTIAL | INSPIRE | They stream-normalize provider quirks; fak controls the native stream |
| 8 | **Finish reason mapping (10+ provider enums → 5 OpenAI standard)** | `src/providers/utils/finishReasonMap.ts:15-144@669825c` | Cross-provider finish reason normalization | ABSENT | INSPIRE | Proxy needs normalization; fak generates native finish reasons |
| 9 | **Strict OpenAI compliance toggle (provider fields when disabled)** | `src/types/requestBody.ts:248-270@669825c` | Optional provider-specific response fields | ABSENT | INSPIRE | Their users want provider features; fak users want reproducibility |
| 10 | **Multi-backend cache strategy (different backends per use case)** | `src/shared/services/cache/index.ts:342-438@669825c` | Purpose-specific cache backends | PARTIAL | INSPIRE | They cache tokens/sessions/config differently; fak has unified cache |
| 11 | **Hook-based guardrails with plugin registry (60+ providers)** | `src/middlewares/hooks/index.ts:246-447@669825c` | Extensible input/output validation | PARTIAL | INSPIRE | They integrate 20+ third-party guardrails; fak has built-in policy |
| 12 | **Fail-open guardrail default (errors → allow)** | `plugins/f5-guardrails/scan.ts:143@669825c` | Availability over security by default | DIVERGENT | WATCH | They prioritize uptime; fak's capability floor is default-deny |
| 13 | **OTLP spans with GenAI semantic conventions** | `src/handlers/services/logsService.ts:97-163@669825c` | Standardized observability | PARTIAL | INSPIRE | They export to external collectors; fak has internal decision journal |
| 14 | **Runtime-aware logging (waitUntil on Workers, async on Node)** | `src/middlewares/log/index.ts:158-164@669825c` | Cross-runtime background execution | PARTIAL | INSPIRE | They deploy to Workers/Node/Deno; fak is Go single-binary |

---

## License Analysis

**License:** MIT (permissive) - `src/LICENSE`  
**Attribution Required:** Yes  
**Compatibility with Apache-2.0 (fak):** ✅ Compatible - MIT code can be adapted into Apache-2.0 with attribution  
**Direct Port Permitted:** Yes, with notice preservation  
**Verdict:** **DIRECT-PORT or ADAPT** permitted for all candidates

---

## Filed Issues

| # | Candidate | Issue | Status |
|---|-----------|-------|--------|
| 2 | Conditional routing DSL | [#10692](https://github.com/anthony-chaudhary/fak/issues/10692) | Filed |
| 3 | Post-hook retry | [#10693](https://github.com/anthony-chaudhary/fak/issues/10693) | Filed |
| 4 | Gateway exception header | [#10694](https://github.com/anthony-chaudhary/fak/issues/10694) | Filed |
| 10 | Multi-backend cache strategy | [#10695](https://github.com/anthony-chaudhary/fak/issues/10695) | Filed |
| 13 | OTLP spans with GenAI conventions | [#10696](https://github.com/anthony-chaudhary/fak/issues/10696) | Filed |
| 11 | Hook-based guardrails plugin architecture | [#10697](https://github.com/anthony-chaudhary/fak/issues/10697) | Filed |
| 1 | Recursive target resolution with config inheritance | [#10698](https://github.com/anthony-chaudhary/fak/issues/10698) | Filed |
| 9 | Strict OpenAI compliance toggle | [#10699](https://github.com/anthony-chaudhary/fak/issues/10699) | Filed |
| 14 | Runtime-aware background logging | [#10700](https://github.com/anthony-chaudhary/fak/issues/10700) | Filed |

**Candidates not filed (DIVERGENT or low fak relevance):**
- #5 Declarative provider config (PARTIAL - fak owns model runtime)
- #6 Dynamic provider config (ABSENT - fak doesn't proxy multi-model providers)
- #7 Two-phase response transforms (PARTIAL - fak controls native stream)
- #8 Finish reason mapping (ABSENT - fak generates native)
- #12 Fail-open guardrail (DIVERGENT - fak is default-deny)

---

## Companion References

- **field-borrow** candidates: All 14 candidates will be witnessed via field-borrow
- **Epic:** Portkey Gateway Borrows (to be created if coherent track emerges)
- **Related studies:** TencentDB Agent Memory (#9851), Hipfire (#10591), Needle (#9852)

---

## Source Evidence

| Class | Evidence |
|-------|----------|
| runtime_source | `git -C $TEMP/study-portkey-gateway rev-parse HEAD` → `669825cbe89ee51569918b8f78a9db486fd69dd4` |
| architecture_design | Deep reads of 8 subsystems via parallel explore agents |
| fak_selfquery_witness | `fak capabilities` queries for: "multi-provider routing", "conditional routing", "guardrails plugin", "cache backend", "OTLP spans", "provider normalization" |
| candidate_matrix | This document, candidate table above |
| license_provenance | `cat $TEMP/study-portkey-gateway/LICENSE` → MIT |
| completeness_critic | This section |
| issue_tracking | [#10692](https://github.com/anthony-chaudhary/fak/issues/10692), [#10693](https://github.com/anthony-chaudhary/fak/issues/10693), [#10694](https://github.com/anthony-chaudhary/fak/issues/10694), [#10695](https://github.com/anthony-chaudhary/fak/issues/10695), [#10696](https://github.com/anthony-chaudhary/fak/issues/10696), [#10697](https://github.com/anthony-chaudhary/fak/issues/10697), [#10698](https://github.com/anthony-chaudhary/fak/issues/10698), [#10699](https://github.com/anthony-chaudhary/fak/issues/10699), [#10700](https://github.com/anthony-chaudhary/fak/issues/10700) | 9 issues independently read back after creation |