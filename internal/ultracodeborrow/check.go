package ultracodeborrow

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"regexp"
	"sort"
	"strings"
	"time"
)

const Schema = "fak.ultracode-workflow-borrow/v1"

var requiredMechanisms = []string{
	"activation",
	"plan-task-graph",
	"child-context",
	"parallelism",
	"messaging",
	"ownership-leases",
	"progress",
	"cancellation",
	"effect-receipts",
	"independent-witness",
	"recovery-resume",
	"usage-quality-observability",
}

var requiredExistingOwners = []int{5964, 5970, 5973, 8156, 8168, 8334, 8337, 8404}

type Artifact struct {
	Schema             string             `json:"schema"`
	Issue              int                `json:"issue"`
	ObservedAt         string             `json:"observed_at"`
	Note               string             `json:"note"`
	Sources            []Source           `json:"sources"`
	LocalTrajectories  []LocalTrajectory  `json:"local_trajectories"`
	ExistingOwners     []ExistingOwner    `json:"existing_owners"`
	Mechanisms         []Mechanism        `json:"mechanisms"`
	Benchmarks         []Benchmark        `json:"benchmarks"`
	UncoveredGaps      []UncoveredGap     `json:"uncovered_gaps"`
	CompletenessCritic CompletenessCritic `json:"completeness_critic"`
}

type Source struct {
	ID             string          `json:"id"`
	Role           string          `json:"role"`
	Kind           string          `json:"kind"`
	Locator        string          `json:"locator"`
	Revision       string          `json:"revision"`
	ObservedAt     string          `json:"observed_at"`
	SourceEventAt  string          `json:"source_event_at"`
	State          string          `json:"state"`
	Platform       string          `json:"platform"`
	RefreshTrigger string          `json:"refresh_trigger"`
	License        LicenseBoundary `json:"license"`
	Provenance     string          `json:"provenance"`
	Anchors        []SourceAnchor  `json:"anchors"`
}

type LicenseBoundary struct {
	SPDX           string `json:"spdx"`
	Boundary       string `json:"boundary"`
	Redistribution string `json:"redistribution"`
	Disposition    string `json:"disposition"`
}

type SourceAnchor struct {
	ID  string `json:"id"`
	Ref string `json:"ref"`
	URL string `json:"url"`
}

type LocalTrajectory struct {
	ID              string         `json:"id"`
	SourceID        string         `json:"source_id"`
	Version         string         `json:"version"`
	ObservedAt      string         `json:"observed_at"`
	SourceEventAt   string         `json:"source_event_at"`
	State           string         `json:"state"`
	StructureDigest string         `json:"structure_digest"`
	Records         int            `json:"records"`
	EventTypes      map[string]int `json:"event_types"`
	ToolCounts      map[string]int `json:"orchestration_tool_counts"`
	Privacy         string         `json:"privacy"`
}

type ExistingOwner struct {
	Issue        int      `json:"issue"`
	Title        string   `json:"title"`
	State        string   `json:"state"`
	MechanismIDs []string `json:"mechanism_ids"`
	Decision     string   `json:"decision"`
}

type Mechanism struct {
	ID                 string     `json:"id"`
	Axis               string     `json:"axis"`
	Status             string     `json:"status"`
	Posture            string     `json:"posture"`
	FactState          string     `json:"fact_state"`
	SourceRefs         []string   `json:"source_refs"`
	FakWitness         FakWitness `json:"fak_witness"`
	LicenseDisposition string     `json:"license_disposition"`
	Owner              Owner      `json:"owner"`
	BenchmarkID        string     `json:"benchmark_id"`
	Decision           string     `json:"decision"`
}

type FakWitness struct {
	ObservedAt string        `json:"observed_at"`
	Queries    []QueryResult `json:"queries"`
	Seams      []string      `json:"seams"`
	Result     string        `json:"result"`
}

type QueryResult struct {
	Command  string   `json:"command"`
	Result   string   `json:"result"`
	TopCards []string `json:"top_cards"`
}

type Owner struct {
	Kind      string `json:"kind"`
	Reference string `json:"reference"`
	Scope     string `json:"scope"`
}

type Benchmark struct {
	ID                     string       `json:"id"`
	MechanismID            string       `json:"mechanism_id"`
	Treatment              string       `json:"treatment"`
	Control                string       `json:"control"`
	Workload               string       `json:"workload"`
	QualityOracle          string       `json:"quality_oracle"`
	Denominators           Denominators `json:"denominators"`
	FalsificationCondition string       `json:"falsification_condition"`
}

type Denominators struct {
	Speed string `json:"speed"`
	Token string `json:"token"`
	Cache string `json:"cache"`
}

type UncoveredGap struct {
	MechanismID string `json:"mechanism_id"`
	Issue       int    `json:"issue"`
	ContractKey string `json:"contract_key"`
	Done        string `json:"done"`
}

type CompletenessCritic struct {
	SourceClassesChecked []string `json:"source_classes_checked"`
	Unavailable          []string `json:"unavailable"`
	Skipped              []string `json:"skipped"`
	Verdict              string   `json:"verdict"`
}

// Invariant: UltraCode borrowing verification is fail-closed and provenance-verified.
// Trailing JSON bytes, unknown fields, or missing schema identities cause immediate rejection.
func Parse(raw []byte) (Artifact, error) {
	var artifact Artifact
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&artifact); err != nil {
		return Artifact{}, fmt.Errorf("decode companion: %w", err)
	}
	if err := dec.Decode(&struct{}{}); err != io.EOF {
		if err == nil {
			return Artifact{}, fmt.Errorf("decode companion: trailing JSON value")
		}
		return Artifact{}, fmt.Errorf("decode companion: trailing data: %w", err)
	}
	if err := Validate(artifact); err != nil {
		return Artifact{}, err
	}
	return artifact, nil
}

// Guard: Candidate mechanisms, immutable source anchors, benchmark falsifiers, and deduplicated
// existing owners must all pass fail-closed verification before adoption.
func Validate(a Artifact) error {
	if a.Schema != Schema || a.Issue != 8484 {
		return fmt.Errorf("identity: schema=%q issue=%d", a.Schema, a.Issue)
	}
	if err := timestamp("observed_at", a.ObservedAt); err != nil {
		return err
	}
	if a.Note != "docs/notes/CONCEPT-STUDY-ULTRACODE-WORKFLOWS-2026-08-21.md" {
		return fmt.Errorf("note: unexpected path %q", a.Note)
	}
	anchors, sourceKinds, err := validateSources(a.Sources)
	if err != nil {
		return err
	}
	if err := validateTrajectories(a.LocalTrajectories, sourceKinds); err != nil {
		return err
	}
	owners, err := validateExistingOwners(a.ExistingOwners)
	if err != nil {
		return err
	}
	benchmarks, err := validateBenchmarks(a.Benchmarks)
	if err != nil {
		return err
	}
	mechanisms, err := validateMechanisms(a.Mechanisms, anchors, owners, benchmarks)
	if err != nil {
		return err
	}
	if err := validateGaps(a.UncoveredGaps, mechanisms, owners); err != nil {
		return err
	}
	if len(a.CompletenessCritic.SourceClassesChecked) == 0 || strings.TrimSpace(a.CompletenessCritic.Verdict) == "" {
		return fmt.Errorf("completeness_critic: source classes and verdict are required")
	}
	return nil
}

func validateSources(sources []Source) (map[string]struct{}, map[string]string, error) {
	if len(sources) < 4 {
		return nil, nil, fmt.Errorf("sources: got %d, want at least 4", len(sources))
	}
	ids := make(map[string]struct{}, len(sources))
	anchors := make(map[string]struct{})
	kinds := make(map[string]string, len(sources))
	alternatives := 0
	for _, source := range sources {
		if blank(source.ID, source.Role, source.Kind, source.Locator, source.Revision, source.State, source.RefreshTrigger, source.Provenance) {
			return nil, nil, fmt.Errorf("source %q: required provenance field is blank", source.ID)
		}
		if _, exists := ids[source.ID]; exists {
			return nil, nil, fmt.Errorf("source %q: duplicate id", source.ID)
		}
		ids[source.ID], kinds[source.ID] = struct{}{}, source.Kind
		if err := timestamp("source "+source.ID+" observed_at", source.ObservedAt); err != nil {
			return nil, nil, err
		}
		if err := timestamp("source "+source.ID+" source_event_at", source.SourceEventAt); err != nil {
			return nil, nil, err
		}
		if blank(source.License.SPDX, source.License.Boundary, source.License.Redistribution, source.License.Disposition) {
			return nil, nil, fmt.Errorf("source %q: missing license boundary", source.ID)
		}
		if !oneOf(source.License.Disposition, "DIRECT-PORT", "ADAPT", "INSPIRE-ONLY", "DO-NOT-USE") {
			return nil, nil, fmt.Errorf("source %q: invalid license disposition %q", source.ID, source.License.Disposition)
		}
		if len(source.Anchors) == 0 {
			return nil, nil, fmt.Errorf("source %q: no immutable anchor", source.ID)
		}
		for _, anchor := range source.Anchors {
			if blank(anchor.ID, anchor.Ref) {
				return nil, nil, fmt.Errorf("source %q: blank anchor", source.ID)
			}
			key := source.ID + ":" + anchor.ID
			if _, exists := anchors[key]; exists {
				return nil, nil, fmt.Errorf("source anchor %q: duplicate", key)
			}
			anchors[key] = struct{}{}
		}
		if source.Role == "alternative" && oneOf(source.Kind, "public-repository", "public-standard") {
			alternatives++
		}
	}
	if alternatives < 2 {
		return nil, nil, fmt.Errorf("sources: got %d public alternatives, want at least 2", alternatives)
	}
	return anchors, kinds, nil
}

func validateTrajectories(rows []LocalTrajectory, sourceKinds map[string]string) error {
	if len(rows) < 1 {
		return fmt.Errorf("local_trajectories: want at least one redacted trajectory")
	}
	ids := make(map[string]struct{}, len(rows))
	for _, row := range rows {
		if blank(row.ID, row.SourceID, row.Version, row.State, row.StructureDigest, row.Privacy) || row.Records <= 0 {
			return fmt.Errorf("local trajectory %q: incomplete structural evidence", row.ID)
		}
		if _, ok := ids[row.ID]; ok {
			return fmt.Errorf("local trajectory %q: duplicate id", row.ID)
		}
		ids[row.ID] = struct{}{}
		if sourceKinds[row.SourceID] != "proprietary-black-box" {
			return fmt.Errorf("local trajectory %q: source %q is not proprietary-black-box", row.ID, row.SourceID)
		}
		if err := timestamp("local trajectory "+row.ID+" observed_at", row.ObservedAt); err != nil {
			return err
		}
		if err := timestamp("local trajectory "+row.ID+" source_event_at", row.SourceEventAt); err != nil {
			return err
		}
		if !strings.HasPrefix(row.StructureDigest, "sha256:") || len(row.EventTypes) == 0 || len(row.ToolCounts) == 0 {
			return fmt.Errorf("local trajectory %q: missing structural digest/counts", row.ID)
		}
	}
	return nil
}

func validateExistingOwners(rows []ExistingOwner) (map[int]struct{}, error) {
	owners := make(map[int]struct{}, len(rows))
	for _, row := range rows {
		if row.Issue <= 0 || blank(row.Title, row.State, row.Decision) || len(row.MechanismIDs) == 0 {
			return nil, fmt.Errorf("existing owner #%d: incomplete", row.Issue)
		}
		if _, exists := owners[row.Issue]; exists {
			return nil, fmt.Errorf("duplicate ownership: existing issue #%d appears twice", row.Issue)
		}
		owners[row.Issue] = struct{}{}
	}
	for _, issue := range requiredExistingOwners {
		if _, ok := owners[issue]; !ok {
			return nil, fmt.Errorf("existing owners: missing required dedupe issue #%d", issue)
		}
	}
	return owners, nil
}

func validateBenchmarks(rows []Benchmark) (map[string]string, error) {
	benchmarks := make(map[string]string, len(rows))
	mechanisms := make(map[string]struct{}, len(rows))
	for _, row := range rows {
		if blank(row.ID, row.MechanismID, row.Treatment, row.Control, row.Workload, row.QualityOracle, row.Denominators.Speed, row.Denominators.Token, row.Denominators.Cache, row.FalsificationCondition) {
			return nil, fmt.Errorf("benchmark %q: treatment/control/workload/oracle/denominators/falsifier are required", row.ID)
		}
		if _, exists := benchmarks[row.ID]; exists {
			return nil, fmt.Errorf("benchmark %q: duplicate id", row.ID)
		}
		if _, exists := mechanisms[row.MechanismID]; exists {
			return nil, fmt.Errorf("benchmark: mechanism %q has more than one row", row.MechanismID)
		}
		benchmarks[row.ID], mechanisms[row.MechanismID] = row.MechanismID, struct{}{}
	}
	return benchmarks, nil
}

func validateMechanisms(rows []Mechanism, anchors map[string]struct{}, existingOwners map[int]struct{}, benchmarks map[string]string) (map[string]string, error) {
	if len(rows) != len(requiredMechanisms) {
		return nil, fmt.Errorf("mechanisms: got %d, want exactly %d", len(rows), len(requiredMechanisms))
	}
	want := make(map[string]struct{}, len(requiredMechanisms))
	for _, id := range requiredMechanisms {
		want[id] = struct{}{}
	}
	got := make(map[string]string, len(rows))
	ownerScopes := make(map[string]struct{}, len(rows))
	for _, row := range rows {
		if _, ok := want[row.ID]; !ok {
			return nil, fmt.Errorf("mechanism %q: not in required matrix", row.ID)
		}
		if _, exists := got[row.ID]; exists {
			return nil, fmt.Errorf("mechanism %q: duplicate", row.ID)
		}
		if blank(row.Axis, row.FactState, row.LicenseDisposition, row.Owner.Kind, row.Owner.Reference, row.Owner.Scope, row.BenchmarkID, row.Decision, row.FakWitness.Result) {
			return nil, fmt.Errorf("mechanism %q: required candidate field is blank", row.ID)
		}
		if !oneOf(row.Status, "PRESENT", "PARTIAL", "ABSENT") || !oneOf(row.Posture, "DEFAULT", "OPTIONAL-MODULE", "RECIPE", "WATCH", "EXCLUDE") {
			return nil, fmt.Errorf("mechanism %q: invalid status/posture %q/%q", row.ID, row.Status, row.Posture)
		}
		if !oneOf(row.FactState, "FACT", "INFERENCE") || !oneOf(row.LicenseDisposition, "DIRECT-PORT", "ADAPT", "INSPIRE-ONLY", "DO-NOT-USE") {
			return nil, fmt.Errorf("mechanism %q: invalid fact/license disposition", row.ID)
		}
		if len(row.SourceRefs) == 0 || len(row.FakWitness.Queries) == 0 || len(row.FakWitness.Seams) == 0 {
			return nil, fmt.Errorf("mechanism %q: source refs, self-query, and fak seams are required", row.ID)
		}
		if err := timestamp("mechanism "+row.ID+" witness observed_at", row.FakWitness.ObservedAt); err != nil {
			return nil, err
		}
		for _, ref := range row.SourceRefs {
			if _, ok := anchors[ref]; !ok {
				return nil, fmt.Errorf("mechanism %q: unknown source anchor %q", row.ID, ref)
			}
		}
		for _, query := range row.FakWitness.Queries {
			if blank(query.Command, query.Result) {
				return nil, fmt.Errorf("mechanism %q: incomplete self-query witness", row.ID)
			}
		}
		if benchmarks[row.BenchmarkID] != row.ID {
			return nil, fmt.Errorf("mechanism %q: benchmark %q missing or points elsewhere", row.ID, row.BenchmarkID)
		}
		ownerKey := row.Owner.Reference + "\x00" + row.Owner.Scope
		if _, exists := ownerScopes[ownerKey]; exists {
			return nil, fmt.Errorf("duplicate ownership: %s scope %q", row.Owner.Reference, row.Owner.Scope)
		}
		ownerScopes[ownerKey] = struct{}{}
		if issue, ok := issueReference(row.Owner.Reference); ok {
			if _, exists := existingOwners[issue]; !exists && row.Owner.Kind != "uncovered-gap" {
				return nil, fmt.Errorf("mechanism %q: owner #%d is neither deduped nor an uncovered gap", row.ID, issue)
			}
		}
		got[row.ID] = row.Status
	}
	return got, nil
}

func validateGaps(rows []UncoveredGap, mechanisms map[string]string, existingOwners map[int]struct{}) error {
	issues := make(map[int]struct{}, len(rows))
	mechanismIDs := make(map[string]struct{}, len(rows))
	for _, row := range rows {
		if row.Issue <= 0 || blank(row.MechanismID, row.ContractKey, row.Done) {
			return fmt.Errorf("uncovered gap: incomplete owner")
		}
		if mechanisms[row.MechanismID] != "PARTIAL" && mechanisms[row.MechanismID] != "ABSENT" {
			return fmt.Errorf("uncovered gap #%d: mechanism %q is not PARTIAL/ABSENT", row.Issue, row.MechanismID)
		}
		if _, duplicate := existingOwners[row.Issue]; duplicate {
			return fmt.Errorf("duplicate ownership: uncovered gap #%d already appears in existing owners", row.Issue)
		}
		if _, duplicate := issues[row.Issue]; duplicate {
			return fmt.Errorf("duplicate ownership: uncovered issue #%d appears twice", row.Issue)
		}
		if _, duplicate := mechanismIDs[row.MechanismID]; duplicate {
			return fmt.Errorf("duplicate ownership: mechanism %q has two uncovered issues", row.MechanismID)
		}
		issues[row.Issue], mechanismIDs[row.MechanismID] = struct{}{}, struct{}{}
	}
	return nil
}

var privateTextPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)\b[a-z]:[\\/](?:users|work|windows|programdata)[\\/]`),
	regexp.MustCompile(`(?i)/(?:users|home)/[^\s/]+/`),
	regexp.MustCompile(`(?i)\\\\[^\s\\]+\\`),
	regexp.MustCompile(`(?i)\b(?:api[_-]?key|access[_-]?token|authorization)\s*[:=]\s*\S+`),
	regexp.MustCompile(`(?i)\bbearer\s+[a-z0-9._~-]+`),
	regexp.MustCompile(`\bsk-[A-Za-z0-9_-]{12,}`),
}

// Invariant: Public documentation and companion JSON must contain no credentials, private paths, or token secrets.
func CheckPublicText(label string, raw []byte) error {
	for _, pattern := range privateTextPatterns {
		if match := pattern.Find(raw); match != nil {
			return fmt.Errorf("%s: private-path/credential pattern %q", label, string(match))
		}
	}
	return nil
}

func timestamp(label, value string) error {
	if _, err := time.Parse(time.RFC3339, value); err != nil {
		return fmt.Errorf("%s: want RFC3339 timestamp: %w", label, err)
	}
	return nil
}

func blank(values ...string) bool {
	for _, value := range values {
		if strings.TrimSpace(value) == "" {
			return true
		}
	}
	return false
}

func oneOf(value string, choices ...string) bool {
	for _, choice := range choices {
		if value == choice {
			return true
		}
	}
	return false
}

func issueReference(ref string) (int, bool) {
	var issue int
	if _, err := fmt.Sscanf(ref, "#%d", &issue); err == nil && issue > 0 {
		return issue, true
	}
	return 0, false
}

func RequiredMechanisms() []string {
	out := append([]string(nil), requiredMechanisms...)
	sort.Strings(out)
	return out
}
