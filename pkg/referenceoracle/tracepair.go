package referenceoracle

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"regexp"
)

// TracePairSchema is the schema id of a retained trace-pair identity document.
const TracePairSchema = "fak.reference_trace_pair.v1"

// ErrInvalidTracePair reports a document that fails the trace-pair contract.
var ErrInvalidTracePair = errors.New("referenceoracle: invalid trace pair")

// TracePair pins the exact artifact/build identity of a retained
// reference-vs-native trace pair: which engine builds produced each side,
// which digests the captured artifacts carry, the capture geometry, and the
// caveats that make those facts interpretable. It is identity data, not trace
// data: the traces themselves live elsewhere and are referenced by digest, so
// a consumer can verify it is comparing against the right chain without
// re-deriving anything.
type TracePair struct {
	Schema string `json:"schema"`
	// Label names the pair (for example "v41-gold-prefill").
	Label string `json:"label,omitempty"`
	// Model names the model family the pair belongs to.
	Model string `json:"model,omitempty"`
	// CapturedUTC is the capture date, YYYY-MM-DD.
	CapturedUTC string           `json:"captured_utc,omitempty"`
	Reference   BuildPin         `json:"reference"`
	Native      BuildPin         `json:"native"`
	ForkRun     ForkRun          `json:"fork_run,omitempty"`
	Prompt      PromptPin        `json:"prompt"`
	Coverage    Coverage         `json:"coverage"`
	Digests     []ArtifactDigest `json:"digests"`
	// AdoptedDigests records that the digests were adopted from the native
	// side's evidence at conversion time instead of being recomputed
	// independently; it requires both a provenance value and at least one
	// caveat.
	AdoptedDigests bool `json:"adopted_digests"`
	// DigestProvenance names how the digests were obtained (for example
	// "adopted_from_native_evidence_at_conversion"); required when
	// AdoptedDigests.
	DigestProvenance string   `json:"digest_provenance,omitempty"`
	Caveats          []string `json:"caveats"`
}

// BuildPin pins one engine build: which source revision ran, which shared
// library or tracer binary it carried, and any probe-only commits layered on
// top that were never published.
type BuildPin struct {
	Repo   string `json:"repo"`
	Branch string `json:"branch,omitempty"`
	Commit string `json:"commit"`
	// ExtraCommits names probe-only commits layered on Commit that were never
	// published; recorded as received from the capture receipt. Empty when
	// Commit alone describes the build.
	ExtraCommits []string `json:"extra_commits,omitempty"`
	// LibSHA256 pins the shared library the build ran with, as a 16-hex
	// sha256 prefix.
	LibSHA256 string `json:"lib_sha256,omitempty"`
	// TracerSHA256 pins the capture tracer binary, as a 16-hex sha256 prefix.
	TracerSHA256 string `json:"tracer_sha256,omitempty"`
}

// ForkRun describes the fork-side capture run. The issuing host is scrubbed
// from public documents; LeaseSuffix keeps only the host-scoped lease id.
type ForkRun struct {
	LeaseSuffix string `json:"lease_suffix,omitempty"`
	RunName     string `json:"run_name,omitempty"`
	Nodes       int    `json:"nodes,omitempty"`
	DurationS   int    `json:"duration_s,omitempty"`
	ExitCode    int    `json:"exit_code,omitempty"`
}

// PromptPin is the exact prompt the pair was captured on.
type PromptPin struct {
	Text          string `json:"text"`
	InputTokenIDs []int  `json:"input_token_ids"`
}

// Coverage is the capture geometry a consumer must reproduce byte-for-byte:
// how many (layer, stage, token) events the reference trace holds, how many
// strided samples per event, and which sample indices are known-excluded.
type Coverage struct {
	Layers          int `json:"layers"`
	Stages          int `json:"stages"`
	Tokens          int `json:"tokens"`
	SamplesPerEvent int `json:"samples_per_event"`
	Events          int `json:"events"`
	Samples         int `json:"samples"`
	// ExcludedSampleIndexes lists sample indices dropped from comparison for
	// a documented semantic reason; each must be < Samples.
	ExcludedSampleIndexes []int `json:"excluded_sample_indexes,omitempty"`
	// NonfiniteRows counts comparison rows whose values are not finite; each
	// row belongs to one event, so the count must be <= Events.
	NonfiniteRows int `json:"nonfinite_rows,omitempty"`
}

// ArtifactDigest pins one captured artifact by name, role and 16-hex sha256
// prefix.
type ArtifactDigest struct {
	Name   string `json:"name"`
	Role   string `json:"role"`
	SHA256 string `json:"sha256"`
}

var hex16 = regexp.MustCompile(`^[0-9a-f]{16}$`)

// ReadTracePair decodes and validates the trace-pair document at path.
func ReadTracePair(path string) (TracePair, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return TracePair{}, err
	}
	return ParseTracePair(data)
}

// ParseTracePair decodes and validates one fak.reference_trace_pair.v1
// document.
func ParseTracePair(data []byte) (TracePair, error) {
	var head struct {
		Schema string `json:"schema"`
	}
	if err := json.Unmarshal(data, &head); err != nil {
		return TracePair{}, fmt.Errorf("%w: %v", ErrInvalidTracePair, err)
	}
	if head.Schema != TracePairSchema {
		return TracePair{}, fmt.Errorf("%w: unknown schema %q", ErrInvalidTracePair, head.Schema)
	}
	var tp TracePair
	if err := json.Unmarshal(data, &tp); err != nil {
		return TracePair{}, fmt.Errorf("%w: %v", ErrInvalidTracePair, err)
	}
	if err := tp.Validate(); err != nil {
		return TracePair{}, err
	}
	return tp, nil
}

func invalidTP(format string, args ...any) error {
	return fmt.Errorf("%w: "+format, append([]any{ErrInvalidTracePair}, args...)...)
}

// Validate checks the structural contract every consumer relies on: named
// reference and native builds, an exact prompt, coverage counts that close
// under their own definitions, in-range excluded sample indices, well-formed
// 16-hex digests, and a digest-provenance caveat whenever the digests were
// adopted rather than independently computed.
func (tp TracePair) Validate() error {
	if tp.Schema != TracePairSchema {
		return invalidTP("schema %q, want %q", tp.Schema, TracePairSchema)
	}
	for i, b := range []BuildPin{tp.Reference, tp.Native} {
		if b.Repo == "" || b.Commit == "" {
			return invalidTP("build pin %d: repo and commit are required", i)
		}
	}
	if tp.Prompt.Text == "" || len(tp.Prompt.InputTokenIDs) == 0 {
		return invalidTP("prompt text and input token ids are required")
	}
	c := tp.Coverage
	if c.Layers <= 0 || c.Stages <= 0 || c.Tokens <= 0 || c.SamplesPerEvent <= 0 {
		return invalidTP("coverage: layers, stages, tokens and samples_per_event must be positive")
	}
	if c.Events != c.Layers*c.Stages*c.Tokens {
		return invalidTP("coverage: events %d != layers*stages*tokens %d", c.Events, c.Layers*c.Stages*c.Tokens)
	}
	if c.Samples != c.Events*c.SamplesPerEvent {
		return invalidTP("coverage: samples %d != events*samples_per_event %d", c.Samples, c.Events*c.SamplesPerEvent)
	}
	for _, idx := range c.ExcludedSampleIndexes {
		if idx < 0 || idx >= c.Samples {
			return invalidTP("coverage: excluded sample index %d out of range [0,%d)", idx, c.Samples)
		}
	}
	if c.NonfiniteRows > c.Events {
		return invalidTP("coverage: %d nonfinite rows exceed %d events", c.NonfiniteRows, c.Events)
	}
	if len(tp.Digests) == 0 {
		return invalidTP("digests are required")
	}
	seen := map[string]bool{}
	for i, d := range tp.Digests {
		if d.Name == "" || d.Role == "" {
			return invalidTP("digests[%d]: name and role are required", i)
		}
		if seen[d.Name] {
			return invalidTP("digests[%d]: duplicate artifact %q", i, d.Name)
		}
		seen[d.Name] = true
		if !hex16.MatchString(d.SHA256) {
			return invalidTP("digests[%d]: %q is not a 16-hex sha256 prefix", i, d.SHA256)
		}
	}
	if tp.AdoptedDigests && (tp.DigestProvenance == "" || len(tp.Caveats) == 0) {
		return invalidTP("adopted_digests requires a digest_provenance value and at least one caveat")
	}
	return nil
}
