// Package edgequal validates the physical low-resource offline witness for issue #8600.
// Invariant: simulators and extrapolated desktop environments are fail-closed rejected.
package edgequal

import (
	"crypto/sha256"
	"embed"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

// Pinned edge qualification constants and model parameters.
const (
	Schema                  = "fak.edgequal.receipt.v1"
	PackVersion             = "issue-8600/v1"
	ModelRepository         = "bartowski/Qwen2.5-1.5B-Instruct-GGUF"
	ModelRepositoryRevision = "d6f592509429a0f25fc337a6d05065356c40d2b2"
	ModelFile               = "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
	ModelSHA256             = "1adf0b11065d8ad2e8123ea110d1ec956dab4ab038eab665614adba04b6c3370"
	Runtime                 = "llama.cpp/server"
	RuntimeRevision         = "2e92ecd0247d25f09797f8fdb044a166522fc05d"
)

//go:embed testdata/pack.json
var files embed.FS

// PackBytes returns the embedded test pack bytes.
func PackBytes() []byte {
	b, err := files.ReadFile("testdata/pack.json")
	if err != nil {
		panic(err)
	}
	return b
}

// PackSHA256 returns the hexadecimal SHA-256 digest of the embedded test pack bytes.
func PackSHA256() string {
	s := sha256.Sum256(PackBytes())
	return hex.EncodeToString(s[:])
}

// Receipt represents a structured execution outcome and verification evidence.
type Receipt struct {
	Schema      string        `json:"schema"`
	Status      string        `json:"status"` // pass or refused
	RefusalCode string        `json:"refusal_code,omitempty"`
	Device      Device        `json:"device"`
	Model       Model         `json:"model"`
	Runtime     RuntimeConfig `json:"runtime"`
	Pack        Artifact      `json:"pack"`
	Execution   Execution     `json:"execution"`
	Metrics     Metrics       `json:"metrics"`
	Cases       []CaseResult  `json:"cases"`
	RawArtifact Artifact      `json:"raw_artifact"`
}

// Device specifies the physical device identity and operating mode.
type Device struct {
	Class                                  string `json:"class"` // android_arm64_phone or laptop_8gib
	Physical                               bool   `json:"physical"`
	Extrapolated                           bool   `json:"extrapolated"`
	Name, OS, SoC, RAM, Storage, PowerMode string
}

// Model specifies the exact model repository, revision, file, digest, and quantization.
type Model struct{ Repository, Revision, File, SHA256, Quantization string }

// RuntimeConfig specifies the execution runtime, revision, template, context window, and thread count.
type RuntimeConfig struct {
	Name, Revision, Template string
	ContextTokens, Threads   int
	Sampling                 string
}

// Artifact represents an immutable, content-addressed artifact reference.
type Artifact struct{ Version, URL, SHA256 string }

// Execution tracks physical execution verification, network isolation, and run duration.
type Execution struct {
	AcquisitionVerified, NetworkDisabled bool
	UndeclaredNetworkCalls               int
	Stage                                string
	DurationSeconds                      int
}

// Metrics records physical device quality, latency percentiles, memory, storage, and energy measurements.
type Metrics struct {
	QualityScore, QualityFloor                                                   float64
	ColdP50MS, ColdP95MS, WarmP50MS, WarmP95MS, PeakRSSMiB, StorageMiB, EnergyWh float64
	ThermalObservation                                                           string
}

// CaseResult holds the verification outcome for a single evaluation case.
type CaseResult struct {
	ID, Language, Tool, OutputSHA256       string
	QualityPass, SchemaPass, InjectionSafe bool
}

var refusalCodes = map[string]bool{"OOM": true, "UNSAFE_TOOL": true, "QUALITY_FLOOR": true, "LATENCY_FLOOR": true, "THERMAL_LIMIT": true}

// Parse decodes a raw JSON receipt, enforcing strict known-field parsing.
func Parse(raw []byte) (Receipt, error) {
	var r Receipt
	d := json.NewDecoder(strings.NewReader(string(raw)))
	d.DisallowUnknownFields()
	if err := d.Decode(&r); err != nil {
		return r, err
	}
	return r, nil
}

// Validate admits only replayable physical evidence. A valid typed refusal is
// evidence of a bounded failure, never a supported/pass claim.
// Guard: enforce fail-closed checks against simulators, extrapolation, and mutable targets.
func Validate(r Receipt) error {
	if r.Schema != Schema {
		return errors.New("EDGEQUAL_SCHEMA")
	}
	if r.Device.Class != "android_arm64_phone" && r.Device.Class != "laptop_8gib" {
		return errors.New("DEVICE_CLASS")
	}
	if !r.Device.Physical {
		return errors.New("SIMULATOR_ONLY")
	}
	if r.Device.Extrapolated {
		return errors.New("DESKTOP_EXTRAPOLATED")
	}
	if empty(r.Device.Name, r.Device.OS, r.Device.SoC, r.Device.RAM, r.Device.Storage, r.Device.PowerMode) {
		return errors.New("DEVICE_IDENTITY_MISSING")
	}
	if r.Device.Class == "laptop_8gib" && r.Device.RAM != "8 GiB" {
		return errors.New("LAPTOP_RAM_NOT_8_GIB")
	}
	if r.Model.Repository != ModelRepository || r.Model.Revision != ModelRepositoryRevision || r.Model.File != ModelFile || r.Model.Quantization != "Q4_K_M" {
		return errors.New("MUTABLE_MODEL_NAME")
	}
	if r.Model.SHA256 != ModelSHA256 || !digest(r.Model.SHA256) {
		return errors.New("MODEL_DIGEST_MISSING")
	}
	if r.Runtime.Name != Runtime || r.Runtime.Revision != RuntimeRevision || !commit(r.Runtime.Revision) {
		return errors.New("MUTABLE_RUNTIME")
	}
	if empty(r.Runtime.Template, r.Runtime.Sampling) || r.Runtime.ContextTokens != 2048 || r.Runtime.Threads < 1 {
		return errors.New("RUNTIME_CONFIG_MISSING")
	}
	if r.Pack.Version != PackVersion || r.Pack.SHA256 != PackSHA256() || !digest(r.Pack.SHA256) {
		return errors.New("PACK_DIGEST_MISSING")
	}
	if !r.Execution.AcquisitionVerified {
		return errors.New("ACQUISITION_UNVERIFIED")
	}
	if !r.Execution.NetworkDisabled || r.Execution.UndeclaredNetworkCalls != 0 {
		return errors.New("UNDECLARED_NETWORK")
	}
	if r.Execution.Stage != "sustained_complete" {
		return errors.New("WEIGHTS_LOADED_ONLY")
	}
	if r.Execution.DurationSeconds < 900 {
		return errors.New("SUSTAINED_RUN_TOO_SHORT")
	}
	if empty(r.RawArtifact.URL) || !digest(r.RawArtifact.SHA256) {
		return errors.New("RAW_ARTIFACT_MISSING")
	}
	if r.Status == "refused" {
		if !refusalCodes[r.RefusalCode] {
			return errors.New("UNTYPED_REFUSAL")
		}
		return nil
	}
	if r.Status != "pass" {
		return errors.New("STATUS_INVALID")
	}
	if r.RefusalCode != "" {
		return errors.New("PASS_HAS_REFUSAL")
	}
	if r.Metrics.QualityScore < r.Metrics.QualityFloor || r.Metrics.QualityFloor <= 0 || r.Metrics.ColdP50MS <= 0 || r.Metrics.ColdP95MS <= 0 || r.Metrics.WarmP50MS <= 0 || r.Metrics.WarmP95MS <= 0 || r.Metrics.PeakRSSMiB <= 0 || r.Metrics.StorageMiB <= 0 || r.Metrics.EnergyWh <= 0 || r.Metrics.ThermalObservation == "" {
		return errors.New("METRICS_FLOOR")
	}
	if len(r.Cases) != 3 {
		return errors.New("LANGUAGE_FIXTURE_INCOMPLETE")
	}
	want := map[string]bool{"hi-hinglish-local-doc": false, "zh-hans-local-doc": false, "en-injection-control": false}
	for _, c := range r.Cases {
		if _, ok := want[c.ID]; !ok {
			return fmt.Errorf("CASE_UNKNOWN: %s", c.ID)
		}
		if !c.QualityPass || !c.SchemaPass || !c.InjectionSafe || !digest(c.OutputSHA256) {
			return fmt.Errorf("CASE_FAILED: %s", c.ID)
		}
		want[c.ID] = true
	}
	for id, ok := range want {
		if !ok {
			return fmt.Errorf("CASE_MISSING: %s", id)
		}
	}
	return nil
}

// ValidatePair verifies that both the phone and laptop receipts are valid and
// executed against the identical model, runtime, and pack fixture.
func ValidatePair(phone, laptop Receipt) error {
	if err := Validate(phone); err != nil {
		return fmt.Errorf("phone: %w", err)
	}
	if err := Validate(laptop); err != nil {
		return fmt.Errorf("laptop: %w", err)
	}
	if phone.Device.Class != "android_arm64_phone" || laptop.Device.Class != "laptop_8gib" {
		return errors.New("PHYSICAL_PAIR_MISSING")
	}
	if phone.Pack.SHA256 != laptop.Pack.SHA256 || phone.Model.SHA256 != laptop.Model.SHA256 || phone.Runtime.Revision != laptop.Runtime.Revision {
		return errors.New("PAIR_NOT_SAME_FIXTURE")
	}
	return nil
}
func digest(s string) bool {
	if len(s) != 64 {
		return false
	}
	_, e := hex.DecodeString(s)
	return e == nil
}
func commit(s string) bool { return len(s) == 40 && digest(s+strings.Repeat("0", 24)) }
func empty(v ...string) bool {
	for _, s := range v {
		if strings.TrimSpace(s) == "" {
			return true
		}
	}
	return false
}
