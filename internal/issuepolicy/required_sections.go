package issuepolicy

import "encoding/json"

// RequiredSectionsSchema identifies the JSON shape of the required-section
// contract emitted by `fak-dev issue contract-sections --json`.
const RequiredSectionsSchema = "fak.issue-required-sections/1"

// RequiredSection describes one issue-body section the runtime gate demands.
type RequiredSection struct {
	Field    string   `json:"field"`
	Headings []string `json:"headings"`
	Required bool     `json:"required"`
	Source   string   `json:"source"`
	Repair   string   `json:"repair,omitempty"`
}

// ProblemFrameRequirement describes the ## Value problem-frame the gate demands.
type ProblemFrameRequirement struct {
	Section        string   `json:"section"`
	CentralityLine string   `json:"centrality_line"`
	CheckLabels    []string `json:"check_labels"`
	Verdicts       []string `json:"verdicts"`
	Repair         string   `json:"repair"`
}

// RequiredSectionsContract is the canonical, drift-proof manifest of the
// required-section set the issue runtime gate enforces. Every entry is derived
// from the live gate (missingRequiredIssueSections, AssessProblemFrame,
// projectWork, AppendProjectWorkDefaults), not hand-maintained, so it cannot
// silently drift from the reviewer.
type RequiredSectionsContract struct {
	Schema               string                  `json:"schema"`
	Sections             []RequiredSection       `json:"sections"`
	ProblemFrame         ProblemFrameRequirement `json:"problem_frame"`
	ProjectWorkSections  []RequiredSection       `json:"project_work_sections"`
	ProductionSections   []RequiredSection       `json:"production_sections"`
	DoneConditionPhrases []string                `json:"done_condition_phrases"`
}

// RequiredSections returns the canonical required-section contract.
func RequiredSections() RequiredSectionsContract {
	return RequiredSectionsContract{
		Schema: RequiredSectionsSchema,
		Sections: []RequiredSection{
			{Field: "current_state", Headings: []string{"Current state", "Today", "Current baseline", "Baseline", "Motivation", "Problem"}, Required: true, Source: "review", Repair: "add ## Current state describing the current baseline the leaf changes"},
			{Field: "scope", Headings: []string{"Scope", "Core through-line", "In scope", "Gold-plating boundary", "Out of scope"}, Required: true, Source: "review", Repair: "add ## Scope, or both a through-line section (## Core through-line / ## In scope) and a boundary section (## Gold-plating boundary / ## Out of scope)"},
			{Field: "done_condition", Headings: []string{"Done condition", "Definition of done", "Acceptance criteria", "Done when", "DoD", "Done condition / witness"}, Required: true, Source: "review", Repair: "add ## Done condition with a checkable done-condition list"},
			{Field: "witness", Headings: []string{"Witness", "Done condition / witness"}, Required: true, Source: "review", Repair: "add ## Witness naming the verifiable command or artifact that proves the done-condition"},
			{Field: "likely_files", Headings: []string{"Likely files", "Path hints", "Paths", "Files", "Likely file", "File scope", "File scopes"}, Required: true, Source: "review", Repair: "add ## Likely files naming the repo paths the leaf will touch"},
			{Field: "scope_class", Headings: []string{"Scope class"}, Required: true, Source: "discoverability", Repair: "declare a done-condition checklist of 1-9 task-list items so the scope class resolves to atomic or compound"},
			{Field: "definition_of_done", Headings: []string{"Definition of done", "Acceptance criteria", "DoD"}, Required: true, Source: "discoverability", Repair: "name a done-condition heading (## Definition of done, ## Acceptance criteria, or DoD) so the checklist is authoritative"},
		},
		ProblemFrame: ProblemFrameRequirement{
			Section:        "Value",
			CentralityLine: "Centrality:",
			CheckLabels:    []string{"P1", "P2", "P3", "P4"},
			Verdicts:       []string{"advanced", "preserved", "N/A"},
			Repair:         "add ## Value with a Centrality: line (Core, Enabling (<named Core outcome>), Stewardship (<obligation>), or Peripheral) and P1-P4 each starting with advanced, preserved, or N/A followed by concrete evidence",
		},
		ProjectWorkSections: []RequiredSection{
			{Field: "work_estimate", Headings: []string{"Work estimate"}, Required: true, Source: "project_work", Repair: "add ## Work estimate with 'N points' (for example: Estimate: 5 points (medium))"},
			{Field: "overall_completion_contribution", Headings: []string{"Overall completion contribution", "Scope contribution"}, Required: true, Source: "project_work", Repair: "add ## Overall completion contribution with 'N/M points' (contribution/parent baseline)"},
			{Field: "completion_standard", Headings: []string{"Completion standard"}, Required: true, Source: "project_work", Repair: "add ## Completion standard using one of " + completionStandardList()},
		},
		ProductionSections: []RequiredSection{
			{Field: "target_operating_envelope", Headings: []string{"Target operating envelope", "Target envelope"}, Required: true, Source: "project_work", Repair: "add ## Target operating envelope describing the target operating envelope for production completion"},
			{Field: "witnessed_operating_envelope", Headings: []string{"Witnessed operating envelope", "Observed operating envelope", "Witnessed envelope"}, Required: true, Source: "project_work", Repair: "add ## Witnessed operating envelope describing the observed operating envelope for production completion"},
		},
		DoneConditionPhrases: []string{"definition of done", "acceptance criteria", "dod"},
	}
}

// DoneConditionPhraseHeadings returns the heading spellings whose presence makes
// the done-condition checklist authoritative (flowmetrics.HasDoD).
func DoneConditionPhraseHeadings() []string {
	return []string{"Definition of done", "Acceptance criteria", "DoD"}
}

// RequiredSectionsJSON returns the canonical contract as indented JSON. It is
// the single encoder shared by the CLI and by any Go caller that needs the
// machine-readable manifest.
func RequiredSectionsJSON() ([]byte, error) {
	return json.MarshalIndent(RequiredSections(), "", "  ")
}
