// Package quantmatrix publishes fak's neutral quantization support registry.
//
// The registry describes interoperability boundaries; it does not rank
// quantization methods or imply quality or performance.
//
// Invariant: quant matrix lookups are fail-closed and deterministic across all capability queries.
// Guard: Unknown capability identifiers or unverified artifact versions
// abstain rather than speculate, while unsupported runtime combinations
// explicitly refuse execution.
package quantmatrix

import (
	"bufio"
	"fmt"
	"sort"
	"strings"
)

type EntryID string
type Class string
type Outcome string
type Code string

const (
	EntryGGUFQ4KCPU            EntryID = "quant.gguf-q4-k.cpu.v1"
	EntryGPTQExternal          EntryID = "quant.gptq.external.v1"
	EntryFP8Research           EntryID = "quant.fp8.research.v1"
	EntryBitsAndBytesInProcess EntryID = "quant.bitsandbytes.in-process.v1"

	ClassNative       Class = "supported"
	ClassExternal     Class = "delegated"
	ClassExperimental Class = "research-only"
	ClassUnavailable  Class = "unsupported"

	OutcomeAllow    Outcome = "allow"
	OutcomeDelegate Outcome = "delegate"
	OutcomeAbstain  Outcome = "abstain"
	OutcomeRefuse   Outcome = "refuse"

	CodeNative         Code = "QUANT_NATIVE"
	CodeExternal       Code = "QUANT_EXTERNAL_RUNTIME"
	CodeExperimental   Code = "QUANT_RESEARCH_ONLY"
	CodeUnknownID      Code = "QUANT_UNKNOWN_ID"
	CodeUnknownVersion Code = "QUANT_UNKNOWN_ARTIFACT_VERSION"
	CodeUnavailable    Code = "QUANT_UNAVAILABLE_COMBINATION"
)

// Entry separates claims about an artifact, its production recipe, the
// runtime that consumes it, and the hardware envelope. ArtifactVersion and
// RuntimeID are matching keys; the longer fields are public claims.
type Entry struct {
	ID              EntryID
	Class           Class
	ArtifactVersion string
	Artifact        string
	Recipe          string
	RuntimeID       string
	Runtime         string
	Hardware        string
	Evidence        string
}

var registry = []Entry{
	{
		ID: EntryGGUFQ4KCPU, Class: ClassNative,
		ArtifactVersion: "gguf-v3", Artifact: "GGUF v3 tensors encoded as Q4_K",
		Recipe:    "Consumes an already-quantized artifact; fak does not claim to produce or calibrate it",
		RuntimeID: "fak-native-cpu", Runtime: "Native fak GGUF loader and CPU Q4_K dequantization; no external runtime",
		Hardware: "Implementation: CPU path. Contract witness: linux/amd64 under WSL; no throughput or quality measurement",
		Evidence: "../internal/ggufload/quant_q4k_loader.go",
	},
	{
		ID: EntryGPTQExternal, Class: ClassExternal,
		ArtifactVersion: "gptq-model", Artifact: "GPTQ-quantized model in the external runtime's accepted artifact layout",
		Recipe:    "Calibration and quantization are owned by the artifact producer, not fak",
		RuntimeID: "external-runtime", Runtime: "Handled by an operator-selected GPTQ-capable runtime; fak has no native GPTQ loader claim",
		Hardware: "Defined by the selected external runtime; not measured or inferred by fak",
		Evidence: "../docs/supported/engines.md",
	},
	{
		ID: EntryFP8Research, Class: ClassExperimental,
		ArtifactVersion: "fak-fp8-e4m3fn", Artifact: "Internal E4M3FN block data used by fak's experimental FP8 compute path; not a portable checkpoint format",
		Recipe:    "Experimental runtime conversion with per-block scaling; no training or calibration quality claim",
		RuntimeID: "fak-research-fp8", Runtime: "Research-only fak FP8 path; not a public serving compatibility promise",
		Hardware: "Implementation includes software/reference paths and optional accelerators; this matrix witness measures none",
		Evidence: "../internal/compute/fp8.go",
	},
	{
		ID: EntryBitsAndBytesInProcess, Class: ClassUnavailable,
		ArtifactVersion: "bnb-4bit", Artifact: "bitsandbytes 4-bit module/checkpoint state",
		Recipe:    "No fak-native bitsandbytes conversion or calibration recipe",
		RuntimeID: "fak-native", Runtime: "Unsupported in-process: fak does not embed the bitsandbytes runtime or silently reinterpret its state",
		Hardware: "No supported fak-native hardware envelope",
		Evidence: "../docs/supported/models.md",
	},
}

// Entries returns a stable copy ordered by capability ID.
func Entries() []Entry {
	out := append([]Entry(nil), registry...)
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

// Lookup returns the registered capability entry for the given capability identifier.
// Invariant: quant matrix lookups are fail-closed and deterministic. Unregistered
// identifiers return an empty Entry and false without fallback speculation.
func Lookup(id EntryID) (Entry, bool) {
	for _, capability := range registry {
		if capability.ID == id {
			return capability, true
		}
	}
	return Entry{}, false
}

type Request struct {
	ID              EntryID
	ArtifactVersion string
	Runtime         string
}

type Decision struct {
	Outcome Outcome
	Reason  Code
}

// Adjudicate evaluates a capability request against registered quantization envelopes.
// Invariant: quant matrix adjudication is fail-closed and deterministic. Unknown capability
// identifiers or mismatched artifact versions return OutcomeAbstain, while unsupported
// combinations return OutcomeRefuse. Adjudicate never guesses a replacement format or runtime.
func Adjudicate(req Request) Decision {
	capability, ok := Lookup(req.ID)
	if !ok {
		return Decision{Outcome: OutcomeAbstain, Reason: CodeUnknownID}
	}
	if req.ArtifactVersion != capability.ArtifactVersion {
		return Decision{Outcome: OutcomeAbstain, Reason: CodeUnknownVersion}
	}
	if req.Runtime != capability.RuntimeID {
		return Decision{Outcome: OutcomeRefuse, Reason: CodeUnavailable}
	}
	switch capability.Class {
	case ClassNative:
		return Decision{Outcome: OutcomeAllow, Reason: CodeNative}
	case ClassExternal:
		return Decision{Outcome: OutcomeDelegate, Reason: CodeExternal}
	case ClassExperimental:
		return Decision{Outcome: OutcomeAbstain, Reason: CodeExperimental}
	default:
		return Decision{Outcome: OutcomeRefuse, Reason: CodeUnavailable}
	}
}

const tableHeader = "| Capability ID | Status | Artifact version | Artifact claim | Recipe claim | Runtime ID | Runtime claim | Hardware envelope | Evidence |"

// ParseDocument independently reads the public Markdown table into typed rows.
// It is intentionally strict so prose cannot drift from the registry unnoticed.
func ParseDocument(document string) ([]Entry, error) {
	scanner := bufio.NewScanner(strings.NewReader(document))
	found := false
	var rows []Entry
	lineNumber := 0
	for scanner.Scan() {
		lineNumber++
		line := strings.TrimSpace(scanner.Text())
		if !found {
			if line == tableHeader {
				found = true
				if !scanner.Scan() {
					return nil, fmt.Errorf("line %d: missing table separator", lineNumber)
				}
				lineNumber++
			}
			continue
		}
		if !strings.HasPrefix(line, "|") {
			break
		}
		fields := splitRow(line)
		if len(fields) != 9 {
			return nil, fmt.Errorf("line %d: got %d columns, want 9", lineNumber, len(fields))
		}
		status := Class(fields[1])
		if !validClass(status) {
			return nil, fmt.Errorf("line %d: unknown status %q", lineNumber, status)
		}
		evidence, err := markdownLinkTarget(fields[8])
		if err != nil {
			return nil, fmt.Errorf("line %d: evidence: %w", lineNumber, err)
		}
		rows = append(rows, Entry{ID: EntryID(fields[0]), Class: status, ArtifactVersion: fields[2], Artifact: fields[3], Recipe: fields[4], RuntimeID: fields[5], Runtime: fields[6], Hardware: fields[7], Evidence: evidence})
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if !found {
		return nil, fmt.Errorf("quantization capability table not found")
	}
	return rows, nil
}

func splitRow(line string) []string {
	parts := strings.Split(strings.Trim(line, "|"), "|")
	for i := range parts {
		parts[i] = strings.TrimSpace(parts[i])
	}
	return parts
}

func markdownLinkTarget(value string) (string, error) {
	open := strings.Index(value, "](")
	if !strings.HasPrefix(value, "[") || open < 2 || !strings.HasSuffix(value, ")") {
		return "", fmt.Errorf("want one Markdown link, got %q", value)
	}
	target := value[open+2 : len(value)-1]
	if target == "" || strings.ContainsAny(target, "#?") {
		return "", fmt.Errorf("want a local file link without fragment or query, got %q", target)
	}
	return target, nil
}

func validClass(status Class) bool {
	return status == ClassNative || status == ClassExternal || status == ClassExperimental || status == ClassUnavailable
}
