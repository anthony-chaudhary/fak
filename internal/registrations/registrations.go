// Package registrations is the "built-in driver list" (the Linux defconfig). It
// blank-imports every leaf that ships enabled in v0.1 so their init() functions
// register against the frozen ABI before the kernel boots. Enabling or disabling
// an idea is exactly one import line here — the kernel itself never imports a
// leaf.
//
// Import order is irrelevant to correctness (the kernel folds by registered
// rank/tier, not import order); it is grouped by subsystem for readability.
package registrations

import (
	// Ref backend + MMU page-out codec (must be present so ActiveResolver works).
	_ "github.com/anthony-chaudhary/fak/internal/blob"

	// Optional DURABLE, on-disk content-addressed store (inert unless FAK_BLOB_DIR
	// is set). Registers a page-out codec under id "blobfs" so a quarantined/cold
	// result can spill to disk and survive a process restart; also the durable tier
	// the storedrv router composes.
	_ "github.com/anthony-chaudhary/fak/internal/blobfs"

	// Optional REMOTE, HTTP object-store content-addressed blob driver (inert
	// unless FAK_BLOB_HTTP_URL is set). Registers a page-out codec under id
	// "blobhttp" — the disaggregated/cloud tier for content that must outlive a
	// single host; pure net/http, no vendor SDK.
	_ "github.com/anthony-chaudhary/fak/internal/blobhttp"

	// Pluggable storage-driver ROUTER: composes the blob/blobfs/blobhttp tiers into
	// one content-addressed namespace and (only when FAK_STORE opts in) becomes the
	// abi RegionBackend so every Ref's bytes route to the tier that fits. Inert
	// unless FAK_STORE is set — blob stays the live backend by default.
	_ "github.com/anthony-chaudhary/fak/internal/storedrv"

	// vDSO fast-path tiers.
	_ "github.com/anthony-chaudhary/fak/internal/vdso"

	// Opt-in audit journal (inert unless FAK_AUDIT_JOURNAL is set).
	_ "github.com/anthony-chaudhary/fak/internal/journal"

	// Pre-flight rung ladder + grammar rung.
	_ "github.com/anthony-chaudhary/fak/internal/grammar"
	_ "github.com/anthony-chaudhary/fak/internal/preflight"
	_ "github.com/anthony-chaudhary/fak/internal/ratelimit"

	// The in-process DOS reference monitor (authoritative adjudicator).
	_ "github.com/anthony-chaudhary/fak/internal/adjudicator"

	// Context-MMU write-time result admission.
	_ "github.com/anthony-chaudhary/fak/internal/ctxmmu"

	// Normalize-and-rescan admitter (rank 5, in front of ctxmmu): closes the
	// obfuscation-evasion gap + provenance-gates trusted-local false positives.
	_ "github.com/anthony-chaudhary/fak/internal/normgate"

	// Information-flow control (CaMeL/FIDES): source-stamps Ref.Taint (rank-20
	// ResultAdmitter) + refuses tainted-data->sensitive-sink flows (rank-30
	// Adjudicator). The provenance complement to the lexical detectors — blocks the
	// paraphrased injection no content gate can catch.
	_ "github.com/anthony-chaudhary/fak/internal/ifc"

	// Require-witness gate backend (in-process dos_verify): corroborates a claimed
	// effect from git evidence the agent did not author before a gated call runs.
	_ "github.com/anthony-chaudhary/fak/internal/witness"

	// Plan control-flow integrity: refuses a tool call that deviates from the
	// operator-approved plan (an injection-derailed gadget) and registers the
	// RequireApproval human-in-the-loop verdict. Inactive until a plan is declared.
	_ "github.com/anthony-chaudhary/fak/internal/plancfi"

	// Ship gate: holds a ship/release call behind the require-witness rung so a
	// claimed ship is git-corroborated (via the witness resolver) before it
	// dispatches. Inactive for non-ship tools (Defers).
	_ "github.com/anthony-chaudhary/fak/internal/shipgate"

	// Git gate: a structural git-shape prefilter (rank 35). Refuses the argv-
	// decidable git hazards in a shell command — force-push, commit --amend, add
	// -A, --no-verify, tag -f, rebase -i — BEFORE the doomed command runs, the in-
	// kernel dual of tools/githooks/*. Defers on non-git calls and on state-
	// dependent laws (OFF_TRUNK). Opt out with FAK_GITGATE=off.
	_ "github.com/anthony-chaudhary/fak/internal/gitgate"

	// AgentDojo ASR steward: the dynamic, ASR-scored replacement for the static
	// poison.json fixture. Self-registers NewASRSteward() into abi.Stewards() so the
	// adaptive attack battery (seeds + generative paraphrases) gates a live build
	// through the full stack — abstains while full-stack ASR==0, fires with a
	// reproducible winning attack the moment a defense regression lets one through.
	// Without this line the leaf's init() never runs and the gate it ships stays dark.
	_ "github.com/anthony-chaudhary/fak/internal/agentdojo"

	// Inference engine drivers (mock = offline echo fallback; cassette/http selectable).
	_ "github.com/anthony-chaudhary/fak/internal/engine"

	// In-kernel model engine: dispatches an allowed tool call to the model fused
	// into the kernel (internal/model). This is now the DEFAULT engine (`--engine
	// inkernel` is implicit); pass `--engine mock` for the offline echo fallback.
	// Lazily built; a deterministic synthetic checkpoint unless FAK_MODEL_DIR names
	// a real export.
	_ "github.com/anthony-chaudhary/fak/internal/modelengine"

	// Stewards (single-invariant validators).
	_ "github.com/anthony-chaudhary/fak/internal/steward"

	// Static tool linter: enrolls the "tool-surface-sound" steward (the booted tool
	// surface must carry no error-severity lint finding — a static answer that
	// swallows a write (TL003), a fast-path serve that bypasses a policy Deny
	// (TL008)). Dormant + abstaining on a clean surface; without this line the
	// invariant ships dark (the leaf's init() never runs). `fak lint` is the same
	// rules run out of band.
	_ "github.com/anthony-chaudhary/fak/internal/headroom"
	_ "github.com/anthony-chaudhary/fak/internal/toollint"
)
