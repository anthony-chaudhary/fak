# Dynamic instruction composition demo captured output

Command, run from the repository root:

```console
go run ./cmd/instructiondemo -selfcheck
```

The block below is the complete deterministic stdout capture consumed by the package regression test.

<!-- BEGIN SELFCHECK OUTPUT -->
```text
{
  "schema_version": "harnesskit-instructions/1",
  "verdict": "PASS",
  "stable_prefix_unchanged": true,
  "full_digest_changed": true,
  "outcome_counts": {
    "invocations": 3,
    "succeeded": 2,
    "failed": 1,
    "by_code": {
      "denied": 1
    },
    "unclassified": 0
  },
  "turns": [
    {
      "schema_version": "harnesskit-instructions/1",
      "prompt_value": [
        {
          "text": "You operate inside fak, an agent kernel: one process that sits between you and the tools you call and adjudicates every tool call before it runs. fak is the irreducible head of your context — these concepts are always resident and never change per turn.",
          "type": "text"
        },
        {
          "text": "The gate: every tool call is adjudicated before it executes. fak denies by structure (a default-deny capability floor you cannot talk past), repairs malformed calls, and quarantines poisoned results. A denied call is refused by structure, not by persuasion.",
          "type": "text"
        },
        {
          "text": "The journal: every decision is appended to a hash-chained, tamper-evident decision journal. A claim counts as true only when a witness in the journal corroborates it; a self-reported success with no witness is not yet done.",
          "type": "text"
        },
        {
          "text": "A capability is a named, versioned affordance — a skill, an MCP tool, or an A2A agent — that the gate may admit. Capabilities are queried by intent, not menu-dumped; their bodies are paged in on demand and evicted under pressure.",
          "type": "text"
        },
        {
          "text": "Policy floor: the deployed capability manifest is default-deny. Any tool call outside the granted allow set is refused. The floor is versioned and resident; it is never relaxed mid-session and never paged out.",
          "type": "text"
        },
        {
          "cache_control": {
            "type": "ephemeral"
          },
          "text": "Safety-critical instructions are always resident — never paged, never compressed, never summarized. Anything load-bearing for the deny floor or fak's identity stays in the spine or policy tier and is excluded from the evictable set by construction.",
          "type": "text"
        },
        {
          "text": "For this turn, focus on correctness.",
          "type": "text"
        }
      ],
      "snapshot": {
        "schema_version": "harnesskit-instructions/1",
        "fragments": [
          {
            "id": "operator-focus",
            "source": "instructiondemo/operator",
            "trust": "application",
            "precedence": 10,
            "lifetime": "turn",
            "audience": [
              "coder"
            ],
            "residency": "ephemeral-tail",
            "content": "For this turn, focus on correctness.",
            "digest": "sha256:4a9b07176ebbe5ac8abce5c282606f66fedccb44969a833842cd1c23b18c9e2f"
          }
        ],
        "digest": "sha256:1f5d3075e9654d6e29943527432d28cdb1d500fe0e1e13402972d8ef8b9728b3",
        "estimated_bytes": 36,
        "estimated_tokens": 9
      },
      "decisions": [
        {
          "id": "operator-focus",
          "included": true,
          "reason": "included by typed instruction snapshot"
        }
      ],
      "digest": "sha256:624778eedf2ae4330f7e431a0df57596acae51918494a355c23d88acbc7cd250",
      "stable_prefix_digest": "blob-sha256:0eab07f1e7352ddc46ea4812761988a4da4bcb71238a3ee434c4c73a219b7169",
      "prefix_audit_status": "ok",
      "estimated_bytes": 1463,
      "estimated_tokens": 366
    },
    {
      "schema_version": "harnesskit-instructions/1",
      "prompt_value": [
        {
          "text": "You operate inside fak, an agent kernel: one process that sits between you and the tools you call and adjudicates every tool call before it runs. fak is the irreducible head of your context — these concepts are always resident and never change per turn.",
          "type": "text"
        },
        {
          "text": "The gate: every tool call is adjudicated before it executes. fak denies by structure (a default-deny capability floor you cannot talk past), repairs malformed calls, and quarantines poisoned results. A denied call is refused by structure, not by persuasion.",
          "type": "text"
        },
        {
          "text": "The journal: every decision is appended to a hash-chained, tamper-evident decision journal. A claim counts as true only when a witness in the journal corroborates it; a self-reported success with no witness is not yet done.",
          "type": "text"
        },
        {
          "text": "A capability is a named, versioned affordance — a skill, an MCP tool, or an A2A agent — that the gate may admit. Capabilities are queried by intent, not menu-dumped; their bodies are paged in on demand and evicted under pressure.",
          "type": "text"
        },
        {
          "text": "Policy floor: the deployed capability manifest is default-deny. Any tool call outside the granted allow set is refused. The floor is versioned and resident; it is never relaxed mid-session and never paged out.",
          "type": "text"
        },
        {
          "cache_control": {
            "type": "ephemeral"
          },
          "text": "Safety-critical instructions are always resident — never paged, never compressed, never summarized. Anything load-bearing for the deny floor or fak's identity stays in the spine or policy tier and is excluded from the evictable set by construction.",
          "type": "text"
        },
        {
          "text": "For this turn, focus on latency.",
          "type": "text"
        }
      ],
      "snapshot": {
        "schema_version": "harnesskit-instructions/1",
        "fragments": [
          {
            "id": "operator-focus",
            "source": "instructiondemo/operator",
            "trust": "application",
            "precedence": 10,
            "lifetime": "turn",
            "audience": [
              "coder"
            ],
            "residency": "ephemeral-tail",
            "content": "For this turn, focus on latency.",
            "digest": "sha256:25d32032a996ccf148c798650bddce0b7af38cc8652c01bcd75c7c12f5066eed"
          }
        ],
        "digest": "sha256:6283ab294c11551dc1eee4977afaf4e5d7220e06aeb2b715155132c3e4ac1e10",
        "estimated_bytes": 32,
        "estimated_tokens": 8
      },
      "decisions": [
        {
          "id": "operator-focus",
          "included": true,
          "reason": "included by typed instruction snapshot"
        }
      ],
      "digest": "sha256:5a21ca3eab329c6c62767b98a0ab30767c88379bf4578b912fec0db7b2de9140",
      "stable_prefix_digest": "blob-sha256:0eab07f1e7352ddc46ea4812761988a4da4bcb71238a3ee434c4c73a219b7169",
      "prefix_audit_status": "ok",
      "estimated_bytes": 1459,
      "estimated_tokens": 365
    }
  ],
  "contract": {
    "schema_version": "harnesskit-instructions/1",
    "resolution": "the host invokes InstructionProvider.Resolve at declared run, thread, or turn boundaries",
    "composition": "providers return typed fragments; the host validates, orders, fingerprints, and realizes them without opaque prompt mutation",
    "ownership": "providers own application fragments; the host owns policy, stable-prefix admission, final serialization, and invocation authority",
    "security": "only host-trusted fragments may enter the stable prefix; untrusted fragments cannot claim positive precedence",
    "determinism": "identical typed inputs and provider output produce byte-stable ordering and SHA-256 snapshot fingerprints",
    "cancellation": "resolution accepts context.Context and returns CodeCanceled with the cancellation cause",
    "compatibility": "same schema_version is additive; changed ordering, authority, or fingerprint semantics require a new schema_version",
    "errors": [
      "invalid",
      "unsupported",
      "conflict",
      "denied",
      "canceled",
      "internal"
    ]
  }
}
```
<!-- END SELFCHECK OUTPUT -->
