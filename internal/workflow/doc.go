// Package workflow is a built-in workflow orchestration layer (D-005, issue #245):
// a small, deterministic DAG engine plus the three patterns agent frameworks reach
// for most — map-reduce, fan-out, and an explicit dependency DAG — expressible as a
// JSON/YAML document and executed CPU-correctly with no model in the loop.
//
// Tier: foundation (1) — see internal/architest. This package may import only
// packages whose tier is <= 1; it imports nothing internal and only the stdlib, so
// it sits cleanly under the agent/gateway integrators that would drive it.
//
// The layer is deliberately abstract over WHAT a task does: the executor schedules
// the DAG, resolves dependencies, retries, and propagates failure, while a pluggable
// Runner performs the actual unit of work (a model call, a tool call, a shell step).
// That separation is what lets the orchestration core be deterministic and
// host-runnable — the acceptance criteria (define a workflow in JSON, run a
// map-reduce, run a fan-out, honor a DAG's dependencies) are all proven without a GPU
// or a network. See workflow.go (the DSL + compiler + patterns) and execute.go (the
// scheduler + fault tolerance).
package workflow
