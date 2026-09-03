---
title: "Study inventory: ruvnet/ruflo"
description: "- Schema: fak-study-inventory-map/1 - Indexed revision: 4dcff483482cee316f47552a961bcbaadc89f378 - Source: https://github.com/ruvnet/ruflo - Observed at:"
---
# Study inventory: ruvnet/ruflo

- **Schema:** `fak-study-inventory-map/1`
- **Indexed revision:** `4dcff483482cee316f47552a961bcbaadc89f378`
- **Source:** https://github.com/ruvnet/ruflo
- **Observed at:** 2026-08-25T00:00:00Z
- **Totals:** 5517 files, 1542 directories, 2315 runtime files, 645 tests/fixtures, 2015 docs, 1613340 text lines

## Source Classes

| Class | Status | Evidence | Note |
|---|---:|---|---|
| `readme_docs` | `covered` | .agents/README.md<br>.claude-plugin/README.md<br>.claude/commands/agents/README.md<br>.claude/commands/analysis/README.md<br>.claude/commands/automation/README.md<br>.claude/commands/coordination/README.md<br>.claude/commands/github/README.md<br>.claude/commands/hive-mind/README.md<br>.claude/commands/hooks/README.md<br>.claude/commands/memory/README.md<br>.claude/commands/monitoring/README.md<br>.claude/commands/optimization/README.md | local tree evidence found |
| `architecture_design` | `covered` | .agents/skills/agent-arch-system-design/SKILL.md<br>.agents/skills/agent-architecture/SKILL.md<br>.agents/skills/v3-ddd-architecture/SKILL.md<br>.claude/agents/architecture/system-design/arch-system-design.md<br>.claude/agents/sparc/architecture.md<br>.claude/commands/sparc/designer.md<br>.claude/helpers/adr-compliance.sh<br>.claude/skills/v3-ddd-architecture/SKILL.md<br>.github/workflows/adr-166-mcp-bridge-security.yml<br>docs/benchmarks/runs/adr-coverage.jsonl<br>plugins/ruflo-adr/agents/adr-architect.md<br>plugins/ruflo-adr/commands/adr.md | local tree evidence found |
| `runtime_source` | `covered` | .agents/config.toml<br>.agents/skills/memory-management/scripts/memory-backup.sh<br>.agents/skills/memory-management/scripts/memory-consolidate.sh<br>.agents/skills/security-audit/scripts/cve-remediate.sh<br>.agents/skills/security-audit/scripts/security-scan.sh<br>.agents/skills/sparc-methodology/scripts/sparc-init.sh<br>.agents/skills/sparc-methodology/scripts/sparc-review.sh<br>.agents/skills/swarm-orchestration/scripts/swarm-monitor.sh<br>.agents/skills/swarm-orchestration/scripts/swarm-start.sh<br>.claude-plugin/hooks/hooks.json<br>.claude-plugin/marketplace.json<br>.claude-plugin/plugin.json | local tree evidence found |
| `tests_fixtures` | `covered` | plugins/ruflo-adr/scripts/__tests__/adr-create-schema-2651.test.mjs<br>plugins/ruflo-adr/scripts/__tests__/index-idempotency-2660.test.mjs<br>plugins/ruflo-adr/scripts/__tests__/parser-bullets-2659.test.mjs<br>plugins/ruflo-adr/scripts/__tests__/skip-brain-dir-2911.test.mjs<br>plugins/ruflo-agntcy/src/casa/__tests__/compile.test.ts<br>plugins/ruflo-agntcy/src/casa/__tests__/enforce.test.ts<br>plugins/ruflo-agntcy/src/receipts/__tests__/casa-receipt.test.ts<br>plugins/ruflo-arena/tests/evolution.test.ts<br>plugins/ruflo-arena/tests/games-arena.test.ts<br>plugins/ruflo-arena/tests/mcp-tools.test.ts<br>plugins/ruflo-arena/tests/tournament.test.ts<br>plugins/ruflo-business-pods/scripts/pod-tick.test.mjs | local tree evidence found |
| `history_changelog_releases` | `covered` | .claude/agents/github/release-manager.md<br>.claude/agents/github/release-swarm.md<br>.claude/commands/github/release-manager.md<br>.claude/commands/github/release-swarm.md<br>.github/workflows/stable-npm-release.yml<br>CHANGELOG.md<br>plugins/ruflo-core/scripts/witness/history.mjs<br>plugins/ruflo-metaharness/scripts/drift-from-history.mjs<br>plugins/ruflo-workflows/commands/gaia-history.md<br>ruflo/docs/adr/ADR-031-HF-CHAT-HISTORY-PERSISTENCE.md<br>ruflo/src/ruvocal/.github/release.yml<br>v3/@claude-flow/cli/.claude/agents/github/release-manager.md | local tree evidence found |
| `open_closed_issues_prs_discussions` | `partial` | .github/ISSUE_TEMPLATE/rollback-incident.md<br>.github/issues/alpha-89-telemetry-implementation.md | local templates found; open/closed issue, PR, and discussion history still require GitHub or forge read-back |
| `roadmap_todos` | `covered` | docs/IMPROVEMENT-ROADMAP.md<br>v3/implementation/migration/v3-migration-roadmap.md<br>v3/implementation/optimization/V3-OPTIMIZATION-ROADMAP.md | local tree evidence found |
| `license_provenance` | `covered` | Cargo.toml<br>LICENSE<br>package.json<br>plugins/ruflo-arena/package.json<br>plugins/ruflo-graph-intelligence/package.json<br>ruflo/package.json<br>ruflo/src/mcp-bridge/package.json<br>ruflo/src/ruvocal/LICENSE<br>ruflo/src/ruvocal/mcp-bridge/package.json<br>ruflo/src/ruvocal/package.json<br>ruflo/src/ruvocal/stub/@reflink/reflink/package.json<br>v3/@claude-flow/aidefence/package.json | local tree evidence found |
| `fak_selfquery_witness` | `external_required` |  | process artifact created by the study pass, not by the foreign repository tree |
| `candidate_matrix` | `external_required` |  | process artifact created by the study pass, not by the foreign repository tree |
| `completeness_critic` | `covered` |  | generated by this inventory map's local tree walk; non-tree omissions remain named in the note |
| `issue_tracking` | `external_required` |  | process artifact created by the study pass, not by the foreign repository tree |

## Subsystems

| Path | Files | Runtime | Tests | Docs | Languages | Examples |
|---|---:|---:|---:|---:|---|---|
| `v3` | 3417 | 1738 | 548 | 968 | TypeScript:1866, Docs:968, Config:257, Shell:69, JavaScript:54, Rust:24 | v3/.agentic-flow/intelligence.json<br>v3/@claude-flow/agents/architect.yaml<br>v3/@claude-flow/agents/coder.yaml<br>v3/@claude-flow/agents/reviewer.yaml<br>v3/@claude-flow/agents/security-architect.yaml<br>v3/@claude-flow/agents/tester.yaml<br>v3/@claude-flow/aidefence/README.md<br>v3/@claude-flow/aidefence/__tests__/threat-detection.test.ts |
| `plugins` | 602 | 151 | 36 | 330 | Docs:330, Config:71, TypeScript:66, Shell:38, JavaScript:1 | plugins/README.md<br>plugins/ruflo-adr/.claude-plugin/plugin.json<br>plugins/ruflo-adr/README.md<br>plugins/ruflo-adr/REFERENCE.md<br>plugins/ruflo-adr/agents/adr-architect.md<br>plugins/ruflo-adr/commands/adr.md<br>plugins/ruflo-adr/docs/adrs/0001-adr-plugin-pattern.md<br>plugins/ruflo-adr/docs/adrs/0002-reconcile-deleted-adrs.md |
| `ruflo` | 556 | 312 | 28 | 50 | TypeScript:267, Config:52, Docs:50, JavaScript:17, Shell:4 | ruflo/README.md<br>ruflo/bin/ruflo.js<br>ruflo/docker-compose.public.yml<br>ruflo/docker-compose.yml<br>ruflo/docs/AUTH.md<br>ruflo/docs/DOCKER.md<br>ruflo/docs/MODELS.md<br>ruflo/docs/TOOLS.md |
| `.claude` | 371 | 37 | 0 | 319 | Docs:319, Shell:28, Config:5, JavaScript:4 | .claude/agents/MIGRATION_SUMMARY.md<br>.claude/agents/analysis/analyze-code-quality.md<br>.claude/agents/analysis/code-analyzer.md<br>.claude/agents/analysis/code-review/analyze-code-quality.md<br>.claude/agents/architecture/system-design/arch-system-design.md<br>.claude/agents/base-template-generator.md<br>.claude/agents/consensus/byzantine-coordinator.md<br>.claude/agents/consensus/crdt-synchronizer.md |
| `docs` | 189 | 0 | 0 | 189 | Config:138, Docs:44 | docs/IMPROVEMENT-ROADMAP.md<br>docs/QUALITY-SWEEP.md<br>docs/STATUS.md<br>docs/TEAM-GATEWAY-CHECKLIST.md<br>docs/USERGUIDE.md<br>docs/_config.yml<br>docs/agenticow/findings.md<br>docs/assets/sv-summit.png |
| `.agents` | 144 | 9 | 0 | 135 | Docs:135, Shell:8, Config:1 | .agents/README.md<br>.agents/config.toml<br>.agents/skills/agent-adaptive-coordinator/SKILL.md<br>.agents/skills/agent-agent/SKILL.md<br>.agents/skills/agent-agentic-payments/SKILL.md<br>.agents/skills/agent-analyze-code-quality/SKILL.md<br>.agents/skills/agent-app-store/SKILL.md<br>.agents/skills/agent-arch-system-design/SKILL.md |
| `scripts` | 105 | 6 | 2 | 0 | Shell:6 | scripts/__tests__/audit-supply-chain.test.mjs<br>scripts/__tests__/stage-internal-runtime-bundles.test.mjs<br>scripts/cleanup-v3.sh<br>scripts/install.sh<br>scripts/smoke-agentbbs.sh<br>scripts/smoke-agenticow.sh<br>scripts/verify-appliance.sh<br>scripts/verify-federation-plugin.sh |
| `.github` | 38 | 32 | 0 | 5 | Config:32, Docs:5 | .github/ISSUE_PATTERN_PERSISTENCE.md<br>.github/ISSUE_TEMPLATE/rollback-incident.md<br>.github/actions/npm-ci-retry/action.yml<br>.github/dependabot.yml<br>.github/issues/alpha-89-telemetry-implementation.md<br>.github/supply-chain/README.md<br>.github/supply-chain/accepted-findings.json<br>.github/supply-chain/allowed-deps.json |
| `tests` | 31 | 0 | 31 | 1 | Shell:16, TypeScript:8, Config:4, Docs:1 | tests/context-persistence-hook.test.mjs<br>tests/docker-regression/Dockerfile<br>tests/docker-regression/Makefile<br>tests/docker-regression/README.md<br>tests/docker-regression/docker-compose.yml<br>tests/docker-regression/fixtures/sample-code.ts<br>tests/docker-regression/fixtures/sample-patterns.json<br>tests/docker-regression/scripts/run-all-tests.sh |
| `.` | 20 | 6 | 0 | 8 | Docs:8, Config:6 | AGENTS.md<br>CHANGELOG.md<br>CLAUDE.local.md<br>CLAUDE.md<br>CONTRIBUTING.md<br>Cargo.toml<br>LICENSE<br>README.md |
| `verification` | 16 | 7 | 0 | 3 | Config:7, Docs:3 | verification/CAPABILITIES.md<br>verification/README.md<br>verification/cli-mcp-tool-baseline.json<br>verification/inventory.json<br>verification/linux/manifest.md.json<br>verification/macos/manifest.md.json<br>verification/mcp-tool-baseline.json<br>verification/results.md |
| `.claude-plugin` | 13 | 7 | 0 | 5 | Docs:5, Shell:4, Config:3 | .claude-plugin/README.md<br>.claude-plugin/docs/INSTALLATION.md<br>.claude-plugin/docs/PLUGIN_SUMMARY.md<br>.claude-plugin/docs/QUICKSTART.md<br>.claude-plugin/docs/STRUCTURE.md<br>.claude-plugin/hooks/hooks.json<br>.claude-plugin/marketplace.json<br>.claude-plugin/plugin.json |
| `plugin` | 4 | 3 | 0 | 0 | Config:2, Shell:1 | plugin/.claude-plugin/plugin.json<br>plugin/hooks/hooks.json<br>plugin/scripts/ruflo-hook.sh |
| `.harness` | 3 | 2 | 0 | 1 | Config:2, Docs:1 | .harness/README.md<br>.harness/manifest.json<br>.harness/mcp-policy.json |
| `bin` | 3 | 3 | 0 | 0 | JavaScript:3 | bin/cli.js<br>bin/npx-repair.js<br>bin/npx-safe-launch.js |
| `data` | 3 | 2 | 0 | 0 | Config:2 | data/clone-data.ledger.json<br>data/clone-data.proof.json |
| `.githooks` | 1 | 0 | 0 | 0 |  |  |
| `services` | 1 | 0 | 0 | 1 | Docs:1 | services/cognitum-analytics/README.md |

## Skipped Directories

- `.git`

## Completeness Critic

local tree inventory walked every non-skipped regular file and grouped immediate subsystems; skipped dependency/cache/control directories: .git; still requires non-tree study artifacts: fak_selfquery_witness, candidate_matrix, issue_tracking
