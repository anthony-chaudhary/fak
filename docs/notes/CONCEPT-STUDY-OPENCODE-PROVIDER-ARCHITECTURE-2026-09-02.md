# OpenCode Provider Architecture Study — 2026-09-02

**Author:** fak agent  
**Source:** https://github.com/anomalyco/opencode (`packages/opencode/src/provider/`, `packages/core/src/catalog.ts`)  
**Pinned revision:** `4eb29a64f0054672950acf789f2b09487ebfbb20`  
**License:** MIT  
**Related issue:** #10724 (dedicated OpenRouter provider contract and roster support)  
**Parent epic:** #8469 (provider contracts)

---

## Executive Summary

This study examines how **OpenCode** (an Effect.ts-based open-source coding agent) structures its AI provider abstraction, compares it with **fak**'s provider contract and model routing architecture, and documents the design decisions for adding dedicated OpenRouter provider support in fak.

---

## 1. OpenCode Provider Architecture

OpenCode integrates with 75+ AI model providers via a unified abstraction layer layered on the Vercel AI SDK (`ai` package) and `models.dev` catalog.

### 1.1 Core Components

| Component | Path | Responsibility |
|---|---|---|
| `Provider.Service` | `packages/opencode/src/provider/provider.ts` | Process-level singleton managing provider lifecycle, state memoization, SDK instantiation, and model resolution. |
| `models.dev` Ingest | `packages/opencode/src/provider/models.ts` | External catalog providing metadata (context tokens, pricing, modalities, capabilities) with three-level fallback (cache -> snapshot -> fetch). |
| `CUSTOM_LOADERS` | `packages/opencode/src/provider/provider.ts` | Dictionary of provider-specific initialization hooks (`autoload`, `options`, `getModel`, `vars`) handling vendor quirks (e.g. Anthropic beta headers, Bedrock AWS credential chains, Vertex ADC auth). |
| `BUNDLED_PROVIDERS` | `packages/opencode/src/provider/provider.ts` | Lazy factory map for bundled AI SDK packages (`@ai-sdk/anthropic`, `@ai-sdk/openai`, etc.). Providers outside the bundle can be dynamically imported. |
| `ProviderTransform` | `packages/opencode/src/provider/transform.ts` | Middleware pipeline normalizing messages, sanitizing Unicode surrogates, stripping unsupported tool schema constructs (`$schema`, `$defs`), injecting prompt cache breakpoints, and mapping reasoning effort. |
| `ProviderAuth` | `packages/opencode/src/provider/auth.ts` | Multi-source authentication resolving API keys and OAuth tokens from env, config, file storage, or plugin hooks. |

### 1.2 Multi-Source Merging Pipeline

OpenCode discovers providers and models through a multi-stage pipeline:
1. **Catalog Load:** Fetch or read cached `models.dev` database.
2. **User Config Override:** Merge `opencode.json` `provider` section (custom base URLs, headers, model blacklists/whitelists).
3. **Environment Discovery:** Scan for known environment variables (`ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, etc.) and register matching providers as `source: "env"`.
4. **Stored Credentials:** Load API keys from persistent auth storage (`opencode auth`).
5. **Plugin Auth:** Execute plugin auth hooks (e.g. Copilot OAuth device code flow).
6. **Custom Loaders:** Run provider-specific loader logic.
7. **Filtering:** Apply disabled provider sets, whitelists, and deprecation filters.

---

## 2. Comparison: OpenCode vs fak Provider Architecture

| Dimension | OpenCode | fak (`internal/modelroute`, `internal/gateway`) |
|---|---|---|
| **Primary Goal** | Broad client-side model access across 75+ hosted vendors. | Agent kernel performance gate (vDSO cache reuse, context compaction) + default-deny security floor (data residency PDP). |
| **Language & Dependencies** | TypeScript / Effect.ts; relies on Vercel AI SDK packages and `models.dev`. | Pure Go; zero external dependencies (stdlib only, architest tier 1). |
| **Provider Contract** | Loose runtime schemas (`Provider.Info`, `Model`) populated by external JSON. | Strict typed `ProviderContract` with explicit `KnowledgeState` (`known`, `unknown`, `not_applicable`) and immutable provenance (`URL`, `Ref`, `Path`, `ObservedAt`, `SHA256`). |
| **Authentication Discipline** | Manages API keys, OAuth refresh, and env sniffing directly. | Credential references (`CredEnv`) only; manifests commit only env-var *names* (e.g. `OPENROUTER_API_KEY`), never secrets. Secret resolution occurs at deferred planner build. |
| **Locality & Residency** | Coarse notion of local models (Ollama via openai-compatible). | Formal `PlacementZone` (`device` / `fleet` / `vendor`); structural `EngineRoute()` prefix (`local:` vs `<kind>:`); fail-closed residency PDP denies remote routes on sensitive/tenant payloads. |
| **Quirks & Transformations** | `ProviderTransform` modifies requests in middleware (surrogate sanitization, schema cleanup). | Route profiles fence unsupported content blocks fail-loud (e.g. DeepSeek Anthropic content fence); cache contracts declare exact TTL and breakpoint semantics. |

---

## 3. Dedicated OpenRouter Support in fak (#10724)

### 3.1 Why Dedicated Support Matters

OpenRouter was previously usable in fak only through the generic `KindOpenAI` wire with a manual `--base-url https://openrouter.ai/api/v1` flag. This left three gaps:
1. **Lack of Identity:** No typed `KindOpenRouter` in the closed `ProviderKind` enum.
2. **Missing Contract:** No canonical `ProviderContract` documenting OpenRouter's API dialect, lack of uniform prompt caching, and retry status codes.
3. **Manual Roster Entry:** No default account or example bindings in `DefaultRoster()`.

### 3.2 Implementation Summary

1. **`internal/modelroute/account.go`:**
   - Added `OpenRouterProviderKey = "openrouter"`, `OpenRouterAPIKeyEnv = "OPENROUTER_API_KEY"`, `OpenRouterOpenAIBaseURL = "https://openrouter.ai/api/v1"`.
   - Added `KindOpenRouter ProviderKind = OpenRouterProviderKey` to the closed `ProviderKind` set.
   - Updated `knownKind()` and `KindBaseURL()` to recognize `KindOpenRouter` and provide its default endpoint.
   - Added `openrouter` account to `DefaultRoster()` with sample bindings (`openrouter-free` -> `openrouter/auto`, `openrouter-best` -> `openrouter/best`).
2. **`internal/modelroute/providercontract.go`:**
   - Added `openRouterProviderContract()` with facts sourced from `@OpenRouterTeam/ai-sdk-provider` and OpenRouter public documentation.
   - Declared `PromptCaching: known(false)` (prompt caching is upstream-dependent, not guaranteed by OpenRouter's router), `CacheTTLSeconds: notApplicable`, `UsageDetails: notApplicable`.
   - Registered OpenRouter in the canonical `providerContracts` slice.
3. **Tests:**
   - Updated `internal/modelroute/providercontract_test.go` and `internal/modelroute/providerprofile_test.go` to validate OpenRouter contract and verify non-caching providers are properly projected.
