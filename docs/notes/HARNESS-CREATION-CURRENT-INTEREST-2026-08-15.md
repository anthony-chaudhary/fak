# Harness-creation current-interest inventory — 2026-08-15

Issue: #6905. Collection window: 2026-08-15 10:35–10:45 America/Los_Angeles.

## Method and limits

Primary evidence was collected reproducibly with `gh api repos/<owner>/<repo>/releases/latest`. Repositories sampled: AG-UI, assistant-ui, CopilotKit, Vercel AI SDK, Mastra, OpenAI Agents Python, goose, Agno, and Dify. This is a purposive release sample, not popularity or market-size evidence.

X was requested explicitly, but this environment has no sanctioned X search/export connector and unauthenticated search is not reproducible enough for quote-level evidence. Therefore **X observations are missing, not inferred**. Issue #6905 stays open until a dated X export or primary-post set is attached. Engagement is omitted because release APIs do not expose comparable social engagement.

## Observed release themes

| Theme | Primary source and date | Observed statement | Product decision |
|---|---|---|---|
| Semantic agent-to-UI protocol | [AG-UI 2026-08-14](https://github.com/ag-ui-protocol/ag-ui/releases/tag/release/2026-08-14) | A new Strands protocol package shipped; agent/UI transport is separately versioned. | Keep fak UI on `fak.harness.run/v1`; never scrape terminal output. #6882 is the spine, #6790 the envelope. |
| UI composition and human feedback | [assistant-ui 0.11.50](https://github.com/assistant-ui/assistant-ui/releases), 2026-08-14 | Notes include cloud thread initialization, feedback payloads, attachment adapters, and React/server compatibility. | Thread identity, feedback/artifacts, adapters, and accessibility belong in #6790, beyond a chat box. |
| Generative UI surface | [CopilotKit 1.68.1](https://github.com/CopilotKit/CopilotKit/releases/tag/v1.68.1), 2026-08-14 | A separately versioned UI/agent product continues rapid release cadence. | Use structured UI and second-skin comparison in #6790/#6808; cadence is not value evidence. |
| Provider interchange | [Vercel AI SDK xAI 3.0.122](https://github.com/vercel/ai/releases/tag/%40ai-sdk/xai%403.0.122), 2026-08-14 | Latest release is a provider package updating provider utilities/OpenAI-compatible dependencies. | Provider adapters/compatibility belong in #6797/#6805; UI must not bind to provider payloads. |
| Local and remote execution | [OpenAI Agents Python 0.7.0](https://github.com/openai/openai-agents-python/releases), 2026-08-13 | Added remote tool execution, session metadata, realtime handoff filtering, and tracing/usage refinements. | Weekend examples need remote tools, durable sessions, handoff boundaries, and observability; ten-minute copy stays offline and bounded. |
| Skills and policy as assets | [goose 1.21.1](https://github.com/block/goose/releases/tag/v1.21.1), 2026-08-14 | Notes include skills, enterprise policy, recipes, extensions, MCP, custom providers/models, and desktop UX. | A strong example is a team pack—skills + policy + model/tool profile + UI—but #6796/#6799 own provenance/permissions. |
| AgentOS composition | [Agno 2.3.24](https://github.com/agno-agi/agno/releases/tag/v2.3.24), 2026-08-14 | Added lifecycle hooks, health reporting, memory/reasoning, and MCP fixes. | Harness creation needs operations/lifecycle, not only prompts; preserve P4 and rebuild receipts. |
| Visual workflow productization | [Dify 1.8.0](https://github.com/langgenius/dify/releases/tag/1.8.0), 2026-08-14 | Release spans workflow nodes, knowledge/retrieval, plugins, observability, and UI. | “This weekend” must name a bounded product, not imply a weekend clone of a full visual platform. |
| Integrated upgrade cost | [Mastra 0.22.2](https://github.com/mastra-ai/mastra/releases), 2026-08-14 | Release spans agent networks, memory/threads, MCP/tools, streaming, observability, and compatibility fixes. | Immutable pins, conformance, rebuilds, and migration evidence stay first class in #6805/#6806. |

## Evidence-derived examples

### Ten-minute bounded support harness

- **For:** a support lead needing consistent read-only triage.
- **Problem:** every agent repeats setup and can drift into unsafe actions.
- **Today:** a prompt template inside a general coding harness.
- **Better because:** an external product owns profile/policy and proves an offline turn.
- **Witness:** init → edit only `product/config.go` → rebuild → selfcheck. The warm build+selfcheck was 10.259 seconds, but a clean-machine ten-minute adoption claim remains `not yet` (#6809).

### Local private native harness

- **For:** a maintainer wanting a fak-native interface rather than another vendor harness.
- **Problem:** terminal-only operation does not prove replaceable product UX.
- **Today:** fak TUI or a third-party harness.
- **Better because:** a loopback web product consumes semantic events and changes branding/layout independently of the kernel.
- **Witness:** `go run ./cmd/harnesswebdemo -selfcheck` plus local browser launch (#6882); full approval/failure/accessibility/second-skin proof remains #6790.

### Weekend team harness pack

- **For:** a team standardizing skills, model/tool access, policy, memory, and presentation.
- **Problem:** copied host files lose provenance, compatibility, permission review, and clean removal.
- **Today:** copy prompts/skills/extensions into every host.
- **Better because:** a content-addressed pack composes explicit capabilities over the stock kernel.
- **Witness:** install, compose, project to two hosts, evaluate, remove, and byte-check restoration; `not yet`, owned by #6796 plus #6792/#6793.

## 2026-08-16 shipped-state refresh

Several local gaps closed after the original collection window:

- The loopback browser now runs a bounded real coding task through fak-native
  Read/Edit/focused-test/diff tools, persists its timeline across restart, and has a
  captured [`gpt-5.6-sol` receipt](../_witnesses/harness-web-demo/LIVE-CODING-GPT-5.6-SOL-2026-08-15.md).
- Four executable starters now cover read-only support, coding workspace, cited research,
  and incident operations; full composable extension remains #6796.
- Released-asset clean-room automation now verifies checksum, extraction, user-owned
  preservation, rebuild, upgrade provenance, and rollback. Publication plus Windows/Linux
  receipts remain #6935.
- The tuned `create-mastra@1.25.0` comparison and participant receipt validator are
  frozen. Eligible independent runs remain zero, so both promotional claims remain
  `not_yet` (#6809, #6911).

The X-specific evidence remains **not yet**. No X connector, authenticated export, or
reproducible primary-source search artifact was available for either collection session.
That is an evidence-boundary result—not evidence that X discussion is absent, unpopular,
or negative. Any X claim still requires a dated artifact with query, account scope,
timestamps, canonical post URLs, and retained raw results.

## Promo status

| Copy | Status | Reason |
|---|---|---|
| “Target: a rebuilt offline harness in ten minutes.” | Witnessed calibration, scoped | Warm and clean maintainer calibrations exist; independent denominator remains zero. |
| “Create a working fak-native harness in 10 minutes.” | **Not yet** | Needs independent clean-machine timing and real-alternative comparison (#6809). |
| “Create your own harness this weekend.” | Modeled target | Bounded examples exist; pack/UI paths have not passed independent conformance. |
| “Use a local fak-native UI today.” | Witnessed bounded coding spine | The loopback UI completed and replayed a real edit/test/diff run; it is not a full IDE or packaged desktop app. |

## Decisions

- Ship the local UI spine (#6882), retaining the exhaustive web-product bar in #6790.
- Keep ten-minute and weekend tracks distinct in `/harness-creator`.
- Treat skills/policy/provider/UI as composable planes (#6792, #6796, #6797, #6799), not a forked monolith.
- Leave #6905 open for missing X primary-source/export evidence and a second-maintainer rerun.
