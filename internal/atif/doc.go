// Package atif projects fak's redacted trajectory Turn corpus onto ATIF — the Agent
// Trajectory Interchange Format — so a fak session round-trips to a portable,
// eval-pipeline-consumable artifact.
//
// # Why
//
// fak captures trajectories as typed per-turn Turn rows (internal/trajectory) and
// exports them as fak's own analysis-shaped JSONL. That JSONL is deliberately
// redacted (verdicts + digests + cost, never prose bodies) and it is fak-private —
// it is not the standard interchange shape a Harbor-class eval / SFT-RL pipeline
// (SWE-bench, LiveCodeBench, Terminal-Bench) reads. ATIF is that shape: one schema
// that normalizes many runtimes' event streams into an ordered list of steps with
// subagent trajectories linked by the parent step that spawned them.
//
// This package is the mapping from fak's Turn corpus onto an ATIF profile
// (schema_version "atif/1"): each Turn becomes one ATIF step, traces become
// trajectories, and a child trace nests under the parent step it was spawned from
// (routed by a parent-id label, the analogue of agent-lens's parent_tool_use_id).
//
// # Redaction is a security boundary
//
// A Turn corpus carries no prose bodies to begin with (fak's redaction is
// structural), so an ATIF export is redacted by construction: it carries the
// analysis metadata, not tool-result text. The one lever an operator has is
// [Options.FullFidelity]: OFF (the default) omits even the human query text and the
// producer label map, emitting only structural identity (tool, verdict, digests,
// cost); ON includes the query and labels. Full-fidelity is therefore a conscious
// operator choice, never the default — the same discipline that governs every fak
// redaction seam. True prose-body fidelity depends on a body-capture layer
// (checkpoint #2394 / sessionledger #2392) that does not exist yet; when it lands,
// this package gains a body field without breaking the schema (fields are additive).
package atif
