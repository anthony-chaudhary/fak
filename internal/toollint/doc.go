// Package toollint is the kernel's STATIC tool linter: it checks the registered
// tool SURFACE for inconsistencies the runtime would otherwise silently paper over
// on every single call.
//
// THE CONCEPT — the definition-time dual of the kernel's call-time re-checks.
// fak's governing stance is "the kernel is the part that doesn't believe the
// agents": it never trusts a tool's self-declared annotations, it RE-CHECKS them
// every call. The vDSO re-derives whether a call is destructive from the tool NAME
// and overrides a lying readOnlyHint (vdso.destructive, unit 32); it re-derives a
// served result's taint instead of trusting the producer (vdso.servedTaint, the
// unit-32 dual); the pre-flight ladder re-validates args against a schema before the
// call fires (preflight.Adjudicate, unit 48). Each of those is a RUNTIME, PER-CALL correction — and
// each is SILENT: the kernel quietly does the safe thing a million times and never
// tells anyone the annotation was wrong.
//
// The tool linter is the missing other half. It runs ONCE, out of band, over the
// tool definitions themselves, and says out loud what the runtime would otherwise
// only ever whisper to itself: this readOnlyHint is dead because the name is
// write-shaped; this pure registration can never be reached under its own hint
// gate; this canned static answer is registered for a tool that mutates the world;
// this schema you show the model is not the schema the kernel enforces. A lint
// finding is the kernel's prediction of its OWN runtime behavior, surfaced at
// definition time so a human can fix the definition instead of shipping a tool
// whose annotation the kernel will fight on every call.
//
// NOT A SYSCALL-PATH DRIVER. The linter is static analysis, not an Adjudicator
// rung — it never runs inside Submit, takes no lease, and changes no verdict. It
// reads the registries the kernel walks (the vDSO fast-path tables, the pre-flight
// schemas) plus, when available, a hint CLASSIFIER (hints are per-call Meta, not
// per-registration, so a kernel-only view cannot see them — the agent/gateway layer
// that maps a tool name to its hints is what makes the hint-shaped rules fire). It
// degrades gracefully: a rule fires only when its input is present, so a partial
// view (FromKernel, registries only) reports the name/schema findings and stays
// silent on the hint findings rather than guessing.
//
// NO DRIFT BY CONSTRUCTION. Every rule that predicts a runtime decision calls the
// SAME predicate the runtime uses — vdso.IsWriteShaped, vdso.ClassifyNamespace —
// never a private copy. If the kernel's heuristic changes, the linter's prediction
// changes with it in the same commit. That is the whole point: the linter is only
// trustworthy if it computes exactly what the kernel will do, so it borrows the
// kernel's own code rather than re-stating it.
package toollint
