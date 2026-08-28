package localappcert

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const CaptureSchema = "fak.local-app-certification-capture/1"

const (
	ReasonCommandExitNonzero = "COMMAND_EXIT_NONZERO"
	ReasonCommandStartFailed = "COMMAND_START_FAILED"
	ReasonCommandTimeout     = "COMMAND_TIMEOUT"
	ReasonCaptureCancelled   = "CAPTURE_CANCELLED"
)

// CaptureSpec describes direct argv commands for one complete certification run.
// RuntimeRevision and Artifact may instead be supplied by CaptureOptions, but a
// value supplied in both places must agree.
type CaptureSpec struct {
	Schema          string
	RuntimeRevision string
	Artifact        string
	Scenarios       []CaptureScenarioSpec
}

type CaptureScenarioSpec struct {
	Name    string
	Command []string
	Timeout string
}

type CaptureOptions struct {
	EvidenceDir     string
	RuntimeRevision string
	Artifact        string
}

// LoadCaptureSpec reads a strict JSON capture specification. Scenario commands
// are a JSON object so the scenario name is declared exactly once at its point
// of use; duplicate object keys are detected rather than silently overwritten.
func LoadCaptureSpec(path string) (CaptureSpec, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return CaptureSpec{}, err
	}
	return ParseCaptureSpec(b)
}

// ParseCaptureSpec parses the capture format without executing any commands.
func ParseCaptureSpec(b []byte) (CaptureSpec, error) {
	var raw struct {
		Schema          string          `json:"schema"`
		RuntimeRevision string          `json:"runtime_revision"`
		Artifact        string          `json:"artifact"`
		Scenarios       json.RawMessage `json:"scenarios"`
	}
	if err := decodeStrict(b, &raw); err != nil {
		return CaptureSpec{}, fmt.Errorf("localappcert: decode capture specification: %w", err)
	}

	scenarios, err := decodeCaptureScenarios(raw.Scenarios)
	if err != nil {
		return CaptureSpec{}, err
	}
	return CaptureSpec{
		Schema:          raw.Schema,
		RuntimeRevision: raw.RuntimeRevision,
		Artifact:        raw.Artifact,
		Scenarios:       scenarios,
	}, nil
}

func decodeCaptureScenarios(b []byte) ([]CaptureScenarioSpec, error) {
	if len(bytes.TrimSpace(b)) == 0 {
		return nil, nil
	}
	dec := json.NewDecoder(bytes.NewReader(b))
	tok, err := dec.Token()
	if err != nil {
		return nil, fmt.Errorf("localappcert: decode capture scenarios: %w", err)
	}
	if tok != json.Delim('{') {
		return nil, errors.New("localappcert: capture scenarios must be an object keyed by required scenario name")
	}
	seen := make(map[string]bool)
	var scenarios []CaptureScenarioSpec
	for dec.More() {
		nameToken, err := dec.Token()
		if err != nil {
			return nil, fmt.Errorf("localappcert: decode capture scenario name: %w", err)
		}
		name, ok := nameToken.(string)
		if !ok {
			return nil, errors.New("localappcert: capture scenario name is not a string")
		}
		if seen[name] {
			return nil, fmt.Errorf("localappcert: duplicate capture scenario %q", name)
		}
		seen[name] = true

		var commandRaw json.RawMessage
		if err := dec.Decode(&commandRaw); err != nil {
			return nil, fmt.Errorf("localappcert: decode capture scenario %q: %w", name, err)
		}
		var command struct {
			Command []string `json:"command"`
			Timeout string   `json:"timeout"`
		}
		if err := decodeStrict(commandRaw, &command); err != nil {
			return nil, fmt.Errorf("localappcert: decode capture scenario %q: %w", name, err)
		}
		scenarios = append(scenarios, CaptureScenarioSpec{Name: name, Command: command.Command, Timeout: command.Timeout})
	}
	if _, err := dec.Token(); err != nil {
		return nil, fmt.Errorf("localappcert: decode capture scenarios: %w", err)
	}
	if err := requireJSONEOF(dec); err != nil {
		return nil, fmt.Errorf("localappcert: decode capture scenarios: %w", err)
	}
	return scenarios, nil
}

func decodeStrict(b []byte, dst any) error {
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	if err := dec.Decode(dst); err != nil {
		return err
	}
	return requireJSONEOF(dec)
}

func requireJSONEOF(dec *json.Decoder) error {
	var extra any
	if err := dec.Decode(&extra); !errors.Is(err, io.EOF) {
		if err == nil {
			return errors.New("multiple JSON values")
		}
		return err
	}
	return nil
}

// ValidateCaptureSpec rejects an incomplete or ambiguous capture before any
// evidence directory is created or command is started.
func ValidateCaptureSpec(spec CaptureSpec, opts CaptureOptions) error {
	if spec.Schema != CaptureSchema {
		return fmt.Errorf("localappcert: capture schema %q, want %q", spec.Schema, CaptureSchema)
	}
	if strings.TrimSpace(opts.EvidenceDir) == "" {
		return errors.New("localappcert: capture evidence directory is required")
	}
	if _, err := resolveMetadata("runtime revision", spec.RuntimeRevision, opts.RuntimeRevision); err != nil {
		return err
	}
	if _, err := resolveMetadata("artifact", spec.Artifact, opts.Artifact); err != nil {
		return err
	}

	required := make(map[string]bool, len(RequiredScenarios))
	for _, name := range RequiredScenarios {
		required[name] = true
	}
	seen := make(map[string]bool, len(spec.Scenarios))
	var duplicates, unknown []string
	for _, scenario := range spec.Scenarios {
		if seen[scenario.Name] {
			duplicates = append(duplicates, scenario.Name)
			continue
		}
		seen[scenario.Name] = true
		if !required[scenario.Name] {
			unknown = append(unknown, scenario.Name)
			continue
		}
		if len(scenario.Command) == 0 || strings.TrimSpace(scenario.Command[0]) == "" {
			return fmt.Errorf("localappcert: capture scenario %q has no command argv", scenario.Name)
		}
		timeout, err := time.ParseDuration(scenario.Timeout)
		if err != nil || timeout <= 0 {
			return fmt.Errorf("localappcert: capture scenario %q has invalid timeout %q", scenario.Name, scenario.Timeout)
		}
	}
	if len(duplicates) != 0 {
		sort.Strings(duplicates)
		return fmt.Errorf("localappcert: duplicate capture scenarios %v", uniqueStrings(duplicates))
	}
	if len(unknown) != 0 {
		sort.Strings(unknown)
		return fmt.Errorf("localappcert: unknown capture scenarios %v", unknown)
	}
	var missing []string
	for _, name := range RequiredScenarios {
		if !seen[name] {
			missing = append(missing, name)
		}
	}
	if len(missing) != 0 {
		return fmt.Errorf("localappcert: missing capture scenarios %v", missing)
	}
	return nil
}

func uniqueStrings(values []string) []string {
	if len(values) < 2 {
		return values
	}
	result := values[:1]
	for _, value := range values[1:] {
		if value != result[len(result)-1] {
			result = append(result, value)
		}
	}
	return result
}

func resolveMetadata(field, specValue, optionValue string) (string, error) {
	specValue = strings.TrimSpace(specValue)
	optionValue = strings.TrimSpace(optionValue)
	if specValue != "" && optionValue != "" && specValue != optionValue {
		return "", fmt.Errorf("localappcert: capture %s conflicts between specification and CLI option", field)
	}
	if optionValue != "" {
		return optionValue, nil
	}
	if specValue != "" {
		return specValue, nil
	}
	return "", fmt.Errorf("localappcert: capture %s is required", field)
}

// Capture runs every required scenario in canonical order. Each command is
// launched directly from argv, without shell interpretation, and all output is
// written even when the command fails or reaches its deadline.
func Capture(ctx context.Context, spec CaptureSpec, opts CaptureOptions) ([]Scenario, error) {
	if err := ValidateCaptureSpec(spec, opts); err != nil {
		return nil, err
	}
	runtimeRevision, err := resolveMetadata("runtime revision", spec.RuntimeRevision, opts.RuntimeRevision)
	if err != nil {
		return nil, err
	}
	artifact, err := resolveMetadata("artifact", spec.Artifact, opts.Artifact)
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(opts.EvidenceDir, 0o755); err != nil {
		return nil, fmt.Errorf("localappcert: create evidence directory: %w", err)
	}

	commands := make(map[string]CaptureScenarioSpec, len(spec.Scenarios))
	for _, scenario := range spec.Scenarios {
		commands[scenario.Name] = scenario
	}
	rows := make([]Scenario, 0, len(RequiredScenarios))
	for index, name := range RequiredScenarios {
		scenario := commands[name]
		timeout, _ := time.ParseDuration(scenario.Timeout)
		commandContext, cancel := context.WithTimeout(ctx, timeout)
		command := exec.CommandContext(commandContext, scenario.Command[0], scenario.Command[1:]...)
		output, commandErr := command.CombinedOutput()
		contextErr := commandContext.Err()
		cancel()

		evidence := filepath.Join(opts.EvidenceDir, fmt.Sprintf("%02d-%s.log", index+1, name))
		if err := os.WriteFile(evidence, output, 0o644); err != nil {
			return rows, fmt.Errorf("localappcert: write evidence for scenario %q: %w", name, err)
		}
		digest := sha256.Sum256(output)
		row := Scenario{
			Name:           name,
			Status:         StatusPass,
			Evidence:       evidence,
			EvidenceSHA256: hex.EncodeToString(digest[:]),
			EvidenceBytes:  int64(len(output)),
			Receipt: &Receipt{
				Engine:          "fak-native",
				Fallback:        "none",
				Artifact:        artifact,
				RuntimeRevision: runtimeRevision,
			},
		}
		if commandErr != nil || contextErr != nil {
			row.Status = StatusFail
			row.Reason = commandFailureReason(commandErr, contextErr)
		}
		rows = append(rows, row)
	}
	return rows, nil
}

func commandFailureReason(commandErr, contextErr error) string {
	if errors.Is(contextErr, context.DeadlineExceeded) {
		return ReasonCommandTimeout
	}
	if errors.Is(contextErr, context.Canceled) {
		return ReasonCaptureCancelled
	}
	var exitErr *exec.ExitError
	if errors.As(commandErr, &exitErr) {
		return ReasonCommandExitNonzero
	}
	return ReasonCommandStartFailed
}
