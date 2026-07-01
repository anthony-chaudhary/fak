package engine

import (
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// VLLMDeterminism is the vLLM served-engine determinism mode fak surfaces as a
// capability (issue #1734). It is an OPERATOR-declared / engine-reported property
// of how the worker was LAUNCHED, NOT something fak can infer from a request's
// sampling parameters. vLLM documents two distinct reproducibility guarantees on
// its public surfaces:
//
//   - batch invariance for online reproducibility (docs/usage/reproducibility.md),
//   - a deterministic offline scheduling option,
//
// and, critically, that WITHOUT one of those, batching variation and numerical
// instability can change outputs even at temperature 0 (docs/usage/faq.md). fak
// therefore treats "unavailable" as the honest default: absent a positive signal
// that the engine is running batch-invariant or deterministic-offline, a
// temperature-0 request is NOT reproducible token-for-token.
//
// This is a correctness/operability bridge, not a kernel rebuild: fak uses the
// engine capability when the operator declares it and labels it "unavailable"
// when they do not -- it does not reimplement batch-invariant kernels in fak.
type VLLMDeterminism string

const (
	// DeterminismUnavailable is the honest default: the engine reports (or the
	// operator has declared) no batch-invariance and no deterministic scheduler,
	// so outputs may vary across batches even at temperature 0.
	DeterminismUnavailable VLLMDeterminism = "unavailable"

	// DeterminismBatchInvariant is vLLM's batch-invariant kernel mode for online
	// reproducibility: identical requests reproduce regardless of co-batched load.
	DeterminismBatchInvariant VLLMDeterminism = "batch_invariance"

	// DeterminismDeterministicOffline is vLLM's deterministic offline scheduling
	// option: a fixed schedule makes an offline run reproducible.
	DeterminismDeterministicOffline VLLMDeterminism = "deterministic_offline_scheduler"
)

// determinismCapPrefix namespaces the determinism capability token under the vLLM
// engine family, matching the existing "engine.vllm.*" tokens (kv-events, metrics).
const determinismCapPrefix = "engine.vllm.determinism."

// ParseVLLMDeterminism normalizes a free-text determinism label (env value, config
// string, or artifact field) onto one of the three canonical modes. Anything it
// does not recognize -- including the empty string -- maps to DeterminismUnavailable,
// so an unknown or missing declaration fails CLOSED to "not reproducible" rather
// than silently asserting determinism.
func ParseVLLMDeterminism(s string) VLLMDeterminism {
	switch strings.ToLower(strings.TrimSpace(strings.ReplaceAll(s, "-", "_"))) {
	case "batch_invariance", "batch_invariant", "invariant":
		return DeterminismBatchInvariant
	case "deterministic_offline_scheduler", "deterministic_offline", "offline", "deterministic":
		return DeterminismDeterministicOffline
	default:
		return DeterminismUnavailable
	}
}

// EnvVLLMDeterminism reads FAK_VLLM_DETERMINISM and normalizes it. Unset or
// unrecognized => DeterminismUnavailable (the fail-closed default). vLLM's
// batch-invariance / deterministic-scheduler posture is a server LAUNCH property,
// so fak reads it from the deployment environment rather than a per-request field.
func EnvVLLMDeterminism() VLLMDeterminism {
	return ParseVLLMDeterminism(os.Getenv("FAK_VLLM_DETERMINISM"))
}

// Normalize returns the mode itself, treating the empty zero value (and any
// unrecognized value) as the fail-closed DeterminismUnavailable so a hand-built
// VLLMDeterminism still reports an honest capability.
func (d VLLMDeterminism) Normalize() VLLMDeterminism {
	switch d {
	case DeterminismBatchInvariant, DeterminismDeterministicOffline:
		return d
	default:
		return DeterminismUnavailable
	}
}

// Capability returns the negotiable capability token for this determinism mode,
// e.g. "engine.vllm.determinism.batch_invariance". Every mode -- including
// "unavailable" -- advertises a token so a consumer always learns the engine's
// declared reproducibility posture rather than having to infer its absence.
func (d VLLMDeterminism) Capability() abi.Capability {
	return abi.Capability(determinismCapPrefix + string(d.Normalize()))
}

// GuaranteesReproducibleOutput reports whether this mode makes identical requests
// reproduce token-for-token. True only for batch-invariance or the deterministic
// offline scheduler; false for "unavailable".
func (d VLLMDeterminism) GuaranteesReproducibleOutput() bool {
	switch d.Normalize() {
	case DeterminismBatchInvariant, DeterminismDeterministicOffline:
		return true
	default:
		return false
	}
}

// TemperatureZeroYieldsDeterminism is the witness guard #1734 requires: it reports
// whether, under this determinism mode, a temperature-0 request actually reproduces
// token-for-token. It is FALSE for DeterminismUnavailable -- temperature 0 alone,
// when the engine reports dynamic batching without batch invariance, does NOT make
// outputs reproducible. A replay/witness claim MUST consult this before citing
// temperature 0 as determinism; a false result means the claim is not witnessed.
func (d VLLMDeterminism) TemperatureZeroYieldsDeterminism() bool {
	return d.GuaranteesReproducibleOutput()
}

// Determinism reports the determinism mode configured for this vLLM worker via the
// environment (FAK_VLLM_DETERMINISM), normalized to the fail-closed default when
// unset. This is the EngineDriver-level determinism accessor.
func (e *VLLMEngine) Determinism() VLLMDeterminism {
	return EnvVLLMDeterminism()
}

// DeterminismCapability reports this vLLM engine's determinism posture as a single
// negotiable capability token (criterion 1 of #1734): one of
// engine.vllm.determinism.{batch_invariance,deterministic_offline_scheduler,unavailable}.
// It is additive to Caps(): a consumer that already reads Caps() can intersect it
// against the three registered determinism tokens, and a consumer that wants only
// the determinism axis calls this directly without scanning the whole capability set.
func (e *VLLMEngine) DeterminismCapability() abi.Capability {
	return e.Determinism().Capability()
}

func init() {
	// Register the three determinism capability tokens as negotiable features so
	// they appear in the capability registry alongside engine.route / engine.openai.
	abi.RegisterCapability(DeterminismUnavailable.Capability())
	abi.RegisterCapability(DeterminismBatchInvariant.Capability())
	abi.RegisterCapability(DeterminismDeterministicOffline.Capability())
}
