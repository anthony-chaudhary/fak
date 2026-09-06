package readmenext

import (
	"encoding/json"
	"fmt"
	"os"
)

// SchemaCandidate is the canonical schema identifier for README candidate fragments.
const SchemaCandidate = "fak-readme-candidate/1"

// Recognized target sections on the README front page.
const (
	TargetHardwareTable     = "hardware_table"
	TargetHeroHeadline      = "hero_headline"
	TargetMemoryOverflow    = "memory_overflow"
	TargetWhyFak            = "why_fak"
	TargetDefaultPriorities = "default_priorities"
	TargetCustom            = "custom"
)

// Recognized retirement actions.
const (
	RetireActionReplaceRow     = "replace_row"
	RetireActionAppendToLegacy = "append_to_legacy"
	RetireActionNone           = "none"
)

// Default repository-relative paths.
const (
	DefaultLegacyArchiveDoc       = "docs/README-legacy.md"
	DefaultHardwareJSONPath       = "docs/benchmarks/hardware-latest.json"
	DefaultReadmePath             = "README.md"
	DefaultBenchmarkAuthorityPath = "BENCHMARK-AUTHORITY.md"
)

// CandidateFragment represents a staged change fragment for the README front page.
type CandidateFragment struct {
	Schema           string        `json:"schema"`
	Issue            int           `json:"issue"`
	Topic            string        `json:"topic"`
	TargetSection    string        `json:"target_section"`
	CandidateContent string        `json:"candidate_content"`
	RetireTarget     RetireTarget  `json:"retire_target"`
	Witness          Witness       `json:"witness"`
	LawsChecklist    LawsChecklist `json:"laws_checklist"`
	ProposedBy       string        `json:"proposed_by,omitempty"`
	Date             string        `json:"date"`
}

// RetireTarget defines how superseded content is handled.
type RetireTarget struct {
	Action           string `json:"action"`                       // "replace_row", "append_to_legacy", "none"
	TargetText       string `json:"target_text,omitempty"`        // Content to be replaced/retired
	LegacyArchiveDoc string `json:"legacy_archive_doc,omitempty"` // Target archive doc, defaults to docs/README-legacy.md
}

// Witness holds the non-forgeable provenance for candidate claims.
type Witness struct {
	AuthorityEntry  string `json:"authority_entry,omitempty"`   // e.g. "BENCHMARK-AUTHORITY.md#issue-10944"
	ReceiptPath     string `json:"receipt_path,omitempty"`      // repo-relative receipt path
	HardwareJSONRow string `json:"hardware_json_row,omitempty"` // platform key e.g. "NVIDIA", "Mac", "AMD"
}

// LawsChecklist records adherence to the three README operator laws.
type LawsChecklist struct {
	SOTAComparison bool `json:"sota_comparison"` // Law 1: SOTA-vs-us, never naive
	FeynmanGloss   bool `json:"feynman_gloss"`   // Law 2: 6th-grade / Feynman gloss
	WideAudience   bool `json:"wide_audience"`   // Law 3: Wide-audience appeal
}

// PublishResult details the outcome of applying candidate fragments.
type PublishResult struct {
	Success             bool     `json:"success"`
	DryRun              bool     `json:"dry_run"`
	ReadmePath          string   `json:"readme_path"`
	LegacyPath          string   `json:"legacy_path,omitempty"`
	HardwareJSONPath    string   `json:"hardware_json_path,omitempty"`
	HardwareJSONUpdated bool     `json:"hardware_json_updated"`
	AppliedFragments    int      `json:"applied_fragments"`
	Changes             []string `json:"changes"`
	RetiredItems        []string `json:"retired_items"`
}

// HardwareLatestManifest matches the schema in docs/benchmarks/hardware-latest.json.
type HardwareLatestManifest struct {
	Schema    string                            `json:"schema"`
	AsOf      string                            `json:"as_of"`
	Platforms map[string]*HardwarePlatformEntry `json:"platforms"`
}

// HardwarePlatformEntry is a single platform record in hardware-latest.json.
type HardwarePlatformEntry struct {
	Observed string `json:"observed"`
	Detail   string `json:"detail"`
	Row      string `json:"row"`
}

// ParseCandidateFragment parses a JSON candidate fragment and sets safe defaults.
func ParseCandidateFragment(data []byte) (*CandidateFragment, error) {
	var f CandidateFragment
	if err := json.Unmarshal(data, &f); err != nil {
		return nil, fmt.Errorf("failed to parse candidate fragment JSON: %w", err)
	}
	if f.RetireTarget.LegacyArchiveDoc == "" {
		f.RetireTarget.LegacyArchiveDoc = DefaultLegacyArchiveDoc
	}
	if f.RetireTarget.Action == "" {
		f.RetireTarget.Action = RetireActionNone
	}
	return &f, nil
}

// LoadCandidateFragmentFile reads and parses a candidate fragment from a file.
func LoadCandidateFragmentFile(filePath string) (*CandidateFragment, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read fragment file %q: %w", filePath, err)
	}
	return ParseCandidateFragment(data)
}
