# Portability formats: reuse the package, registry, sync, and context layers

**Date:** 2026-08-13  
**Issue:** [#6596](https://github.com/anthony-chaudhary/fak/issues/6596)  
**Parent:** [#6589](https://github.com/anthony-chaudhary/fak/issues/6589)  
**Status:** public research note; sources pinned below

## Verdict

fak should not invent one universal package format. The interoperable design is a
layered profile:

1. keep an object's native payload where a public contract already exists (Agent
   Skills for skills, MCP `server.json` for MCP servers, ATIF for trajectories);
2. carry collections as digest-addressed OCI 1.1 artifacts and use registry
   referrers for attestations, signatures, SBOMs, and revocation statements;
3. borrow TUF's channel and anti-rollback semantics, with Sigstore bundles for
   public identity and pinned operator keys for private/offline domains;
4. borrow chezmoi/Nix/Git's source-versus-rendered-target, lock, preview, and
   three-way reconciliation model; and
5. let fak own the part no package standard can: typed sensitivity, capability
   admission, precedence, transactional activation, receipt, rollback, and the
   process lifecycle handshake.

This yields portable bytes without delegating execution trust to a registry. OCI
digests prove content identity, not publisher identity. Signatures prove an
identity authorized a digest, not that its scripts are safe. Installation must
therefore remain inert through inspection and become executable only through a
separate, witnessed fak activation transaction.

## Value frame and problem checks

- **Centrality:** Enabling. This unblocks the managed-context portability program
  without moving fak's core tool-call checkpoint.
- **For:** operators moving a mixed agent workspace between tools, machines, and
  organizations.
- **Problem:** today's useful state is split among tool-specific directories,
  registries, policy stores, and transcript formats; copying it loses provenance,
  precedence, secrets posture, and rollback.
- **Today:** Git and ad hoc archives transport bytes, while each consumer invents
  its own activation semantics.
- **Better because:** native formats remain usable by their ecosystems and fak
  adds one inspect/plan/activate/receipt boundary instead of another universe of
  package managers.
- **Witness:** the pinned matrices, repository read-backs, and issue-sized
  recommendations in this note.
- **P1 managed context:** preserve object identity and dependencies without
  loading all payloads into every turn.
- **P2 net-true efficiency:** reuse existing parsers, registries, mirrors, caches,
  and signatures; do not build a proprietary registry or CRDT engine first.
- **P3 bounded adaptation:** explicit compatibility, sensitivity, precedence, and
  capability checks precede activation.
- **P4 integrated operations:** preview, signed provenance, offline cache,
  rollback, revocation, and lifecycle acknowledgements are one transaction trail.

## Method

The first pass queried fak's committed documentation index with
`go run ./cmd/fak-dev index docs` for `agent package portability`, `skill registry
manifest`, `config sync rollback`, `session context export`, `plugin marketplace`,
`mcp registry`, `portable session image`, and `organization policy precedence
signing`. It found the existing capability index, MCP publication, plugin-marketplace,
portable-session, and signed-org-policy seams before the external comparison. The
first five pins and the matrix headings were published on
[#6596](https://github.com/anthony-chaudhary/fak/issues/6596#issuecomment-5284367336)
before deep study.

All external facts below come from project specifications, project documentation,
or pinned upstream source. A tag is paired with its resolved commit so a mutable web
page is not the evidence boundary. `PRESENT`, `PARTIAL`, and `ABSENT` describe the
candidate seam in fak, not the upstream project's maturity.

## Pinned primary sources

| System | Pin used | License at the pin | Why it is in the study |
|---|---|---|---|
| Agent Skills | [`agentskills/agentskills@69ef37e`](https://github.com/agentskills/agentskills/commit/69ef37e9424c0a7ea9dd2293b559e43ec8176379), 2026-08-09 | Apache-2.0 code; specification/docs note CC-BY-4.0 | Native skill directory and `SKILL.md` contract |
| Skills over MCP (experimental) | [`modelcontextprotocol/experimental-ext-skills@967120e`](https://github.com/modelcontextprotocol/experimental-ext-skills/commit/967120e732bfbedde9f9e3c6e082e00db40b1eb5), 2026-08-11 | Apache-2.0 | Draft remote projection of skill files; explicitly not an official MCP extension |
| MCP Registry | [`modelcontextprotocol/registry@a25f166`](https://github.com/modelcontextprotocol/registry/commit/a25f166b4b5bee06eeecb75e4f37b2a44a8aa5be), 2026-08-10 | Apache-2.0 for new code/spec contributions; legacy files retain their notices; docs generally CC-BY-4.0 | `server.json`, namespace ownership, versions, status, and downstream synchronization |
| chezmoi | [`v2.72.0` / `f81cb32`](https://github.com/twpayne/chezmoi/tree/f81cb321789aa3df62871248f5e4d361a59e7cc1), 2026-08-02 | MIT | Source/target split, templates, password-manager references, preview, three-way merge |
| Nix | [`2.35.2` / `2c73b59`](https://github.com/NixOS/nix/tree/2c73b59da29606068c0c98db015dd3a66955525d), 2026-08-12 | LGPL-2.1 | Immutable store, locked dependency graph, substituters, generations |
| OCI Image Spec | [`v1.1.1` / `147f9c1`](https://github.com/opencontainers/image-spec/tree/147f9c13cedb47a0c4d9a11a222961073d585877), 2025-02-24 | Apache-2.0 | Descriptor, digest, manifest, artifact type, subject |
| OCI Distribution Spec | [`v1.1.1` / `a139cc4`](https://github.com/opencontainers/distribution-spec/tree/a139cc423184af6078077b9b7ee336eddbd03f8f), 2025-01-23 | Apache-2.0 | Registry push/pull and referrers/fallback discovery |
| Tekton Pipelines | [`v1.15.0` / `70f7308`](https://github.com/tektoncd/pipeline/tree/70f730815249a12aa725266fba7f3ff39653aaea), 2026-07-31 | Apache-2.0 | Workflow Tasks/Pipelines carried as OCI bundles |
| yadm | [`3.5.0` / `7eabaee`](https://github.com/yadm-dev/yadm/tree/7eabaee84c8bd9521e56966e5c88e7a435fdd9c7), 2025-03-03 | GPL-3.0 | Direct Git dotfiles, scored alternates, encrypted archive, bootstrap hook |
| Sigstore cosign | [`v3.1.3` / `11926fa`](https://github.com/sigstore/cosign/tree/11926fa5bbbbde47e88fc006b625a17769b743b2), 2026-08-05 | Apache-2.0 | Key/keyless signatures and offline-verifiable verification bundles |
| TUF specification | [`v1.0.36` / `59e601e`](https://github.com/theupdateframework/specification/tree/59e601ed29c0d2e497264ae8b31c11b8ef07df1e), 2026-08-10 | Community Specification License 1.0 | Threshold roots, targets, snapshots, timestamp, expiry, delegation, anti-rollback |
| Automerge JS | [`automerge-3.4.1` / `8d7b12f`](https://github.com/automerge/automerge/tree/8d7b12f8da553afbb325e37a6c66942b8dd4d994), 2026-08-12 | MIT | CRDT sync and deterministic preservation of concurrent conflicts |
| OPA | [`v1.19.0` / `1e32c79`](https://github.com/open-policy-agent/opa/tree/1e32c796e8979b1bda2f768138500b1deb95ff24), 2026-07-30 | Apache-2.0 | Signed policy/data bundles, discovery, last-good activation, status |
| Go | [`go1.26.0` / `d90b98e`](https://github.com/golang/go/tree/d90b98e65320778f3b1f99a6951ab20f04d218b3), 2026-02-10 | BSD-3-Clause | Public source-package identity, dependency graph, proxy, checksum, vendor/cache |
| Harbor ATIF | [`harbor-framework/harbor@ac398bb`](https://github.com/harbor-framework/harbor/blob/ac398bbda7c4c1073461797d3b95c2455cc671b5/rfcs/0001-trajectory-format.md), ATIF v1.7, 2026-08-13 source revision | Apache-2.0 | Portable, complete agent trajectory JSON including subagent relationships |
| OpenTelemetry semantic conventions | [`v1.44.0` / `e10a930`](https://github.com/open-telemetry/semantic-conventions/tree/e10a930844c6951757a43b849d364f7d056ac32b), 2026-08-04 | Apache-2.0 | Interoperable telemetry attributes and GenAI spans, not resumable context |

The local Git comparison used `git version 2.49.0.windows.1` and the versioned
[`git-merge` documentation](https://git-scm.com/docs/git-merge/2.49.0). Git is
GPL-2.0; fak interoperates through the executable and repository format rather than
copying Git implementation code.

## Coverage matrix: identity, trust, and activation

`—` means the upstream format deliberately leaves the concern to another layer.

| Candidate | Identity | Dependency | Precedence | Secret handling | Provenance | Signing | Compatibility | Executable-content trust |
|---|---|---|---|---|---|---|---|---|
| Agent Skills | Directory/name plus description; name must match the parent directory | No dependency graph | Host-defined | No secret contract | Optional license and metadata, no immutable source identity | None | Free-text `compatibility`; unknown metadata may extend it | `scripts/` and experimental `allowed-tools` exist; the host decides whether and how to execute |
| Skills over MCP draft | `skill://` resources plus an index | Index enumerates files, not semantic dependencies | Client-defined | No package-level secret contract | Remote server is a source; draft tells clients to inspect provenance | None | Experimental extension revision only | Remote skills are explicitly untrusted and must not imply local execution |
| MCP Registry / `server.json` | Reverse-DNS server name plus version | Runtime packages/remotes, not an object dependency solver | Registry status/version selection; client chooses | Environment arguments can be marked secret; value delivery is client-side | Repository and publisher/namespace evidence | Package hash/digest and namespace authentication, not a general publisher signature | Schema version, package type, transport, active/deprecated/deleted status | Package command and environment are launch inputs; client must inspect and sandbox |
| Tekton Bundles | OCI reference plus Kubernetes `apiVersion`, `kind`, and object name | Pipelines refer to Tasks and parameters | Explicit resolver parameters and cluster policy | Kubernetes Secret reference for registry credentials | OCI digest and Kubernetes metadata | External registry/Sigstore policy | API version/kind | Tasks and Pipelines are executable workflow definitions; Kubernetes admission remains decisive |
| chezmoi | Source-state attribute/path to target path | Templates, includes, scripts, external password-manager references | Later data sources override earlier ones; templates render a target | Password-manager templates and encrypted source support keep values out of ordinary source | Optional Git history and source-state metadata | Git signing is optional, not a chezmoi package signature | Template conditions and OS-specific data | Scripts are visible source but executable during apply; preview is necessary |
| yadm | Git path/ref maps directly to a home path | Git submodules and bootstrap conventions | Alternate files are scored by host/user/class/distro/OS conditions | GPG/OpenSSL encrypted archive | Git history | Optional Git signatures | Alternate filename conditions | Executable bootstrap after clone is an unacceptable implicit-install default for fak |
| Nix | Flake reference locks to revision/content hash; results live at immutable store paths | Explicit locked input graph | Module/evaluator rules, priorities, and overrides | Secrets must be injected outside the world-readable store | Lock data, derivation, and content-addressed store | Binary-cache signatures; flake locks are integrity, not universal publisher identity | System/features and evaluator semantics | Derivation builders execute, normally sandboxed; adopting Nix would also adopt its evaluator |
| OCI Image + Distribution | Descriptor `{mediaType,digest,size}`; tags are mutable aliases | Descriptor DAG and optional `subject`; no semantic package constraints | Client chooses tag/digest and artifact profile | Registry credentials are outside the artifact | Annotations plus digest graph | None in core; signatures are separate referrer artifacts | Media types, artifact type, platform | Content is opaque until a profile/runtime interprets it |
| Sigstore + TUF | Signed subject digest; TUF role/key IDs and target paths | TUF target/delegation metadata | Threshold role hierarchy and delegated target paths | Private keys and OIDC credentials are out of band | Certificate/issuer/transparency proof or pinned-key metadata | Yes: bundled signature/certificate/transparency material; threshold metadata | Versioned metadata and client policy | Verification authorizes bytes; it does not make scripts safe |
| Git + Automerge | Git object/commit IDs; Automerge actor/change graph | Git submodules only; no package constraints | Branch/config policy or application schema | Filters/encryption/key management are external | Commit DAG and authorship metadata | Optional signed commits/tags | Application content schema | Git hooks/submodules can execute; CRDT data is inert until the application interprets it |
| OPA bundles | Bundle name, manifest revision, roots | Policy/data roots, not general package dependencies | Discovery bootstrap and non-overlapping roots determine activation | Service credentials stay outside the bundle | Manifest/revision and status report | Optional detached bundle signatures with out-of-band verification keys | Rego version and manifest constraints | Policy code is evaluated only after verification; OPA does not ship the control plane |
| Go modules | Module path plus semantic/pseudo version | `go.mod` graph with minimal-version selection, replace, exclude, retract | Main module and explicit directives govern selection | Proxy/VCS credentials and `GOPRIVATE` are external | VCS origin plus `go.sum`/checksum database integrity | Checksum transparency is not publisher identity | Module major path and `go` version | Fetch has no npm-style lifecycle hook, but build/test/install compiles and can run code |
| Harbor ATIF v1.7 | `trajectory_id`, optional `session_id`, agent identity, step IDs | Subagent trajectories and continued trajectories | Event order; no package precedence | Full-fidelity content is allowed; redaction policy is producer-specific | Agent, tool, observation, timestamp, usage, and artifact fields | None | Exact `schema_version: ATIF-v1.7` | JSON is inert, though recorded commands/tool calls may be sensitive |
| OpenTelemetry GenAI conventions | Trace/span/conversation IDs and resource identity | Parent/links, not package dependencies | Resource/scope/span attribute rules | Collection/sampling/redaction policy is deployment-specific | Trace/resource attributes and timestamps | OTLP transport identity is external | Semantic-convention stability/version | Telemetry is inert evidence, not an activation or replay object |

## Coverage matrix: state movement and distribution

| Candidate | Merge | Rollback | Revocation | Offline | Federation |
|---|---|---|---|---|---|
| Agent Skills / Skills over MCP | VCS/file merge only | VCS or host-specific | No standard | Local directories work offline | Copy/VCS; experimental MCP projection can expose remote files |
| MCP Registry / `server.json` | None | Pin an older version | `active`, `deprecated`, `deleted` state; installed-copy behavior is client-defined | Package cache is implementation-specific | Incremental API supports downstream registry synchronization; namespaces use GitHub/OIDC/DNS/HTTP ownership proofs |
| Tekton Bundles | Kubernetes/Git merge outside the bundle | Pin an earlier digest | Registry/policy concern | Resolver cache modes exist | Any compatible OCI registry |
| chezmoi | Explicit destination/source/target three-way merge | Source-control rollback and managed apply | No distribution revocation | Local source and cache work offline | Any supported VCS/source service |
| yadm | Git merge | Git checkout/revert | No distribution revocation | Local clone works offline | Git remotes |
| Nix | Declarative evaluation, not user-facing three-way merge | Generations and locked inputs | Remove/replace an input; no universal security-revocation channel | Store, lock, cache, and vendored inputs support offline use | Multiple substituters and source fetchers |
| OCI Image + Distribution | None | Retain/pin a prior digest | Delete/tombstone/channel policy is outside the content spec | OCI layout and local cache | Any conformant registry; referrers have a specified fallback scheme |
| Sigstore + TUF | Metadata roles compose; content merge is out of scope | Version/freeze/rollback checks | Root rotation, expiry, delegated target removal/replacement | Sigstore bundles verify offline; TUF uses cached metadata within validity policy | TUF mirrors/delegations and OCI registries |
| Git + Automerge | Git three-way merge; Automerge preserves concurrent CRDT changes and same-field conflicts | Git revert/reset or retained Automerge heads | No built-in distribution revocation | First-class local operation | Git remotes or CRDT peer protocol |
| OPA bundles | Snapshot/delta update, not human source merge | A failed new bundle leaves the last active bundle running | Poll and replace; no TUF-style root/revocation role | Persisted last-good bundle | HTTP/object-store/control-plane implementations |
| Go modules | Source merge is Git's concern | Pin/replace an older version | `retract` is advisory; proxies may retain content | Module cache and vendor mode | Proxy chain plus direct VCS |
| Harbor ATIF v1.7 | Append/continued/subagent relationships, not concurrent state merge | Retain an earlier export | None | Standalone JSON | Any producer/consumer that implements the schema |
| OpenTelemetry GenAI conventions | Collectors join traces/links, not resumable state | Retention/replay is backend-specific | None | Local buffering is optional | OTLP and multi-vendor collectors/backends |

## fak seam coverage

| Concern | Status | Repository witness | Gap that remains for portability |
|---|---|---|---|
| Identity | **PRESENT** | `internal/capindex/capindex.go` defines `CapRef{Kind,Name,Version}`, digest-bearing `CapCard`, and a protocol-blind `Capability`; `internal/capindex/catalog.go` reconciles resolver indexes | One canonical object/collection ID profile and adapter-native ID mapping are still #6598 work |
| Dependency | **PARTIAL** | `CapCard.RequiredCaps` records runtime capability requirements; OCI descriptors and Go modules already exist at adjacent seams | No portable package dependency/lock resolver or cycle/conflict report |
| Precedence | **PARTIAL** | `internal/policy/orgprecedence.go` and `internal/policy/orgcompose.go` implement central/operator/default policy ordering | No single ordered fold across local, collection, org, environment, and session objects |
| Secret handling | **PARTIAL** | `internal/policy/policy.go` has typed secret posture/patterns; `internal/policy/secretfloor_test.go` witnesses inherited-secret stripping; snapshot artifacts scrub credential shapes | Object/field sensitivity classification, export egress, and secret references are intentionally #6600 |
| Provenance | **PARTIAL** | Capability and session-image digests, snapshot metadata, policy issuers, and ATIF events record local provenance | No general publisher/source/transform chain tied to a portable collection receipt |
| Signing | **PARTIAL** | `internal/policy/orgenvelope.go` verifies Ed25519 envelopes; `orgledger.go` persists anti-rollback identity/version state | No general package signature/referrer verification profile or public OIDC identity path |
| Compatibility | **PARTIAL** | Schema tags, `CapRef.Version`, session migration audit, and min-version policy gates exist | No adapter compatibility report spanning host, model, platform, dependencies, and unknown fields |
| Merge | **ABSENT** | Git can merge repository files, but no committed portability package implements typed collection reconciliation | #6601 needs base/source/target preview, tombstones, conflict classes, and a receipt; CRDT is not a semantic substitute |
| Rollback | **PARTIAL** | Session-image restore, policy last-good fallback, capability residency/swap, and Git history are independent rollback seams | No atomic mixed-collection activation receipt that reverses all affected stores |
| Revocation | **PARTIAL** | Runtime capability refusal and org-policy expiry/anti-rollback exist | No installed-package tombstone, channel revocation, replacement, or offline revocation policy |
| Offline | **PARTIAL** | Local skills, `.faksession` images, snapshots, Git, and `internal/policy/orgpull.go`'s checksummed last-good cache work offline | No two-machine offline package/channel reconciliation witness |
| Federation | **PARTIAL** | fak publishes its MCP server through `server.json` and `docs/fak/mcp-registry.md`; Git and Go proxy/remotes are usable | No general collection artifact profile, namespace mapping, or federated org/public catalog |
| Executable-content trust | **PARTIAL** | `internal/policy/skill.go` compiles skill frontmatter into span-scoped capability admission and screens unsafe bodies; the adjudicator remains default-deny | Install/inspect and activate/execute are not yet a universal package transaction, especially for scripts and workflow bundles |

### Existing format-specific read-backs

- **Agent Skills payload is already present.** `internal/capindex/skill_resolver.go`
  indexes local skill directories and faults their content by `CapRef`; it should be
  extended by an adapter profile, not replaced with a fak-only skill language.
- **MCP registry publication is already present.** Root `server.json` and
  `docs/fak/mcp-registry.md` describe the official registry package. This proves one
  publisher path, not a general collection installer.
- **Portable resume integrity is already present.** `internal/sessionimage/sessionimage.go`
  writes a logical `.faksession` bundle with SHA-256 part integrity and re-arms the
  quarantine gate on rehydration. It is a resume/checkpoint format, not a public
  interchange schema or signed registry artifact.
- **Enterprise policy distribution is substantially present.** The org envelope,
  durable ledger, precedence fold, fetcher, expiry, min-version check, and checksummed
  last-good cache implement the important OPA/TUF-adjacent behavior for one issuer.
  A Rego evaluator or OPA control plane would duplicate this unless a real integration
  supplies demand.
- **ATIF interoperability is only partial.** `internal/atif/atif.go` emits
  `schema_version: "atif/1"` and a fak-specific top-level bundle. Harbor ATIF v1.7
  requires `schema_version: "ATIF-v1.7"` and its trajectory/agent/step shape. The
  exporter shipped for #3440 is useful, but it cannot honestly claim v1.7 conformance
  without a compatibility adapter and golden validation.

## Adopt, adapter, inspire, and reject decisions

| Decision | Candidate seam | fak status | Rationale and boundary | License/security constraint |
|---|---|---|---|---|
| **ADOPT** | Agent Skills directory as the native skill payload | **PRESENT**, packaging **PARTIAL** | Preserve `SKILL.md`, references, assets, and scripts byte-for-byte; add fak envelope metadata for digest, source, dependencies, sensitivity, compatibility, and activation receipt | Apache/CC-BY permits implementation and documentation use; remote scripts remain inert until explicit capability admission |
| **ADAPTER / MONITOR** | Experimental Skills over MCP | **ABSENT** | Its resource/index projection is a useful adapter shape, but the repository explicitly says it is experimental and not an official MCP extension | Do not make a draft URI/layout the canonical stored identity; treat the remote server as untrusted |
| **ADOPT** | MCP `server.json` for MCP servers only | **PRESENT** publisher, installer **ABSENT** | Reuse its service identity, package/remotes, repository, and lifecycle status; do not stretch it into a universal workspace manifest | Registry ownership/hash is not a sufficient executable trust decision |
| **ADOPT** | OCI 1.1 artifact/distribution as collection carrier | **PARTIAL** | Define a small fak media-type profile over descriptors, manifests, layers, subjects, and referrers; activate by digest, never by an unresolved mutable tag | Apache-2.0; registry auth and a digest alone do not prove publisher identity |
| **ADOPT** | TUF channel semantics plus Sigstore/private-key verification profiles | **ABSENT** for general packages | Reuse threshold roots, targets/snapshot/timestamp, expiry, delegation, anti-rollback, and offline bundles instead of inventing update security | TUF spec license differs from implementation licenses; Sigstore keyless identity needs issuer/subject policy and offline verification material |
| **INSPIRE** | chezmoi + Nix + Git transaction model | **PARTIAL** primitives | Canonical source, rendered target, immutable pin, preview, base/source/target reconciliation, receipt, and rollback compose with #6601 | Do not place secrets in Nix-like world-readable stores; never run yadm-style bootstrap implicitly |
| **REJECT for activation merge** | Automerge/CRDT as the first sync substrate | **ABSENT** | CRDTs resolve structural concurrency, not fak's semantic dependency, precedence, sensitivity, or capability conflicts. Start with explicit typed three-way merge; a future CRDT may carry fields whose merge law is proven | MIT is compatible; security risk is semantic auto-acceptance, not the library license |
| **ADAPTER only on demand** | OPA bundle distribution | Native equivalent **PRESENT**, adapter **ABSENT** | Map verified OPA policy/data into the org-policy boundary only when an operator needs OPA interoperability; do not embed Rego or replace the shipped envelope/precedence plane | Apache-2.0; signed policy still requires constrained roots and fail-closed activation |
| **ADOPT narrowly** | Go modules for compiled Go extension source | **PRESENT** | Reuse module identity, dependency selection, proxy/cache, checksum, and vendor behavior in the source-build marketplace plane; it is not the format for sessions, policies, or skills | BSD-3-Clause; checksum integrity is not author identity; compilation remains a controlled build action |
| **ADOPT / FIX ADAPTER** | Harbor ATIF v1.7 for trajectory interchange | **PARTIAL** | Emit/ingest standards-shaped ATIF for analysis and handoff; retain `.faksession` for resumable private runtime state | Apache-2.0; exports need explicit redacted/full-fidelity modes because the schema can carry prompts, observations, and commands |
| **ADAPTER, not storage** | OpenTelemetry GenAI conventions | **ABSENT** | Project ATIF/runtime events into spans and logs for observability; never claim an OTLP trace can restore a session | Apache-2.0; telemetry export is another egress boundary |
| **REJECT** | One proprietary `.fakpkg` schema that replaces all native formats | **ABSENT** and should remain so | A thin collection envelope may name typed native objects, but copying their payload into a fak-only schema would lose upstream compatibility and create a new registry/parser ecosystem | Concentrates parser, update, and executable-content risk in one unreviewed format |

Tekton's bundle resolver is a positive proof that workflow objects can retain their
native schema while OCI supplies distribution. yadm's bootstrap is the corresponding
negative proof: transport completion must never imply executable activation.

## Composition with #6432 and #6272

- [#6432](https://github.com/anthony-chaudhary/fak/issues/6432) owns live
  process-forest identity, request/acknowledgement, quiesce, checkpoint, restore,
  crash reconciliation, and adapter negotiation. A portability install or switch
  must call that lifecycle transaction; it must not invent a second stop/resume
  protocol. [#6607](https://github.com/anthony-chaudhary/fak/issues/6607) is the
  existing integration point.
- [#6272](https://github.com/anthony-chaudhary/fak/issues/6272) owns canonical term,
  contrast, owner, and freshness disambiguation. The Object / Collection / Context /
  Package / Channel / Transaction vocabulary should register there; this note does
  not create a parallel terminology index.
- [#6598](https://github.com/anthony-chaudhary/fak/issues/6598) owns the neutral
  object/collection schema, [#6600](https://github.com/anthony-chaudhary/fak/issues/6600)
  owns sensitivity and egress, and [#6601](https://github.com/anthony-chaudhary/fak/issues/6601)
  owns offline synchronization. The recommendations below are adapter/channel-sized
  contracts beneath those issues, not replacements for them.

## Supported recommendations

Only findings with a concrete repository seam and an independently shippable witness
become backlog. The five checked items below are the supported follow-ups. Four are
newly filed contracts; the three-way-sync finding routes to the already exact #6601
contract instead of duplicating it.

1. **ADOPT Agent Skills through an inert adapter profile.** Preserve the native
   directory, define exact envelope mappings and unknown-field round trip, and prove
   scripts cannot run during inspect/install. Follow-up:
   [#6626](https://github.com/anthony-chaudhary/fak/issues/6626).
2. **ADOPT an OCI artifact profile and bridge MCP `server.json` only for MCP-service
   objects.** Prove an OCI-layout/registry round trip by digest with referrer discovery
   and no implicit activation. Follow-up:
   [#6627](https://github.com/anthony-chaudhary/fak/issues/6627).
3. **ADOPT TUF channel metadata with Sigstore and pinned-key verification profiles.**
   Prove expiry, threshold root rotation, rollback refusal, delegated namespace,
   tombstone/replacement, and offline last-good behavior. Follow-up:
   [#6628](https://github.com/anthony-chaudhary/fak/issues/6628).
4. **INSPIRE sync from chezmoi/Nix/Git and keep CRDT out of semantic activation.**
   Prove typed base/source/target conflicts, tombstones, preview, receipt, rollback,
   and deterministic two-home reconciliation before considering a CRDT carrier.
   Existing exact contract:
   [#6601](https://github.com/anthony-chaudhary/fak/issues/6601).
5. **ADOPT Harbor ATIF v1.7 at the trajectory boundary while retaining `.faksession`
   for resume.** Prove external-schema validation, subagent relationships,
   version negotiation, and redacted/full-fidelity export. Follow-up:
   [#6629](https://github.com/anthony-chaudhary/fak/issues/6629).
6. **ADAPTER OPA only when a named integration needs it.** Existing org-policy
   envelope, precedence, pull, last-good, and anti-rollback seams already cover the
   local problem. No issue is filed from this study because no unmet operator demand
   was found.
7. **ADOPT Go modules only for compiled Go source extensions; reject package-manager
   lifecycle scripts and universal dynamic plugins.** This composes with the existing
   marketplace [#3807](https://github.com/anthony-chaudhary/fak/issues/3807), skill
   loader [#1103](https://github.com/anthony-chaudhary/fak/issues/1103), and version
   identity [#2461](https://github.com/anthony-chaudhary/fak/issues/2461). No duplicate
   issue is filed.

## What remains a finding, not backlog

- A CRDT could later help with schema fields that have an explicit commutative merge
  law. There is no evidence yet that full collection activation has that property.
- Nix can remain an optional adapter/export target for operators already using it.
  Requiring its evaluator would be a larger dependency and trust decision than the
  portability problem justifies.
- Tekton Bundles demonstrate the OCI pattern. A Tekton adapter has no demonstrated fak
  user demand, so it is not filed.
- OpenTelemetry projection is useful after the ATIF boundary is conformant. It does not
  independently unblock package, restore, or synchronization portability.
- OPA bundle ingestion is justified only by a concrete enterprise integration. fak's
  existing signed org-policy plane should not be displaced speculatively.

## Reproduction checklist

1. Check the source revision and date in the pin table, then open the linked commit or
   tag rather than relying on a search-result summary.
2. From a clean archive of fak's committed tip, run the self-query terms in **Method**
   with `go run ./cmd/fak-dev index docs`.
3. Read the repository witnesses named in **fak seam coverage** and verify the ATIF
   constants independently: fak currently says `atif/1`; the pinned Harbor RFC says
   `ATIF-v1.7`.
4. Read back every filed follow-up and confirm its title/body carries a deduplication
   key, parent/research links, done conditions, and a witness rather than only an idea.
