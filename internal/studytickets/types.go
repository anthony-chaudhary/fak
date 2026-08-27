// Package studytickets closes a bounded study-priority queue against a captured
// issue corpus without contacting or mutating GitHub.
package studytickets

import (
	"errors"

	"github.com/anthony-chaudhary/fak/internal/studyprio"
)

const (
	Schema       = "fak.study-ticket-closure/1"
	ReportSchema = "fak.study-ticket-closure-report/1"
	ParentIssue  = 9268
)

var ErrInvalid = errors.New("studytickets: invalid ticket closure")

type BuildOptions struct {
	PriorityPath       string
	JoinPath           string
	ForgePath          string
	AdjacencyPath      string
	ClassificationPath string
}

type ValidateOptions struct {
	BuildOptions
	LedgerPath string
	ReportPath string
}

type SourceReceipt struct {
	Path          string `json:"path"`
	SHA256        string `json:"sha256"`
	Schema        string `json:"schema"`
	Revision      string `json:"revision,omitempty"`
	Cutoff        string `json:"cutoff,omitempty"`
	RecordCount   int    `json:"record_count,omitempty"`
	ReceiptSHA256 string `json:"receipt_sha256,omitempty"`
	IndexChecksum string `json:"index_checksum,omitempty"`
	CaptureStatus string `json:"capture_status,omitempty"`
}

type Sources struct {
	Priority  SourceReceipt `json:"priority"`
	Join      SourceReceipt `json:"join"`
	Forge     SourceReceipt `json:"forge"`
	Adjacency SourceReceipt `json:"adjacency"`
}

type DispositionCount struct {
	Disposition string `json:"disposition"`
	Count       int    `json:"count"`
	Actionable  int    `json:"actionable"`
}

type Coverage struct {
	JoinClusters           int                `json:"join_clusters"`
	ActionableClusters     int                `json:"actionable_clusters"`
	DispositionCounts      []DispositionCount `json:"disposition_counts"`
	UncoveredActionable    int                `json:"uncovered_actionable"`
	QueueSelections        int                `json:"selected_candidates"`
	MappedSourceClusters   int                `json:"mapped_source_clusters"`
	SelectedUnmapped       int                `json:"selected_unmapped"`
	Unclassified           int                `json:"unclassified"`
	UnmappedActionable     int                `json:"unmapped_actionable"`
	ClosureLeftovers       int                `json:"closure_leftovers"`
	CreatedCount           int                `json:"created_count"`
	ReusedCount            int                `json:"reused_count"`
	ConstructionDefinition string             `json:"construction_definition"`
}

type Ticket struct {
	CandidateID        string                    `json:"candidate_id"`
	WorkItemTitle      string                    `json:"candidate_title"`
	Issue              int                       `json:"issue"`
	URL                string                    `json:"url"`
	State              string                    `json:"state"`
	Title              string                    `json:"title"`
	RecordSHA256       string                    `json:"record_sha256"`
	CreatedAt          string                    `json:"created_at"`
	UpdatedAt          string                    `json:"updated_at"`
	Labels             []string                  `json:"labels"`
	Horizon            string                    `json:"horizon"`
	QueueRank          int                       `json:"queue_rank"`
	Score              int                       `json:"score"`
	Centrality         string                    `json:"centrality"`
	Dependencies       []string                  `json:"dependencies"`
	SourceClusters     []studyprio.SourceMapping `json:"source_clusters"`
	PurposeBuilt       bool                      `json:"constructed_for_candidate"`
	ReusedExistingWork bool                      `json:"reused_existing_work"`
	NativeConstraint   string                    `json:"native_constraint"`
}

type QueueEntry struct {
	Rank         int      `json:"rank"`
	CandidateID  string   `json:"candidate_id"`
	Issue        int      `json:"issue"`
	Horizon      string   `json:"horizon"`
	Dependencies []string `json:"dependencies"`
}

type AdjacencyClass struct {
	Repository string `json:"repository"`
	Class      string `json:"class"`
	Status     string `json:"status"`
	Notes      string `json:"notes"`
}

type AdjacencyReceipt struct {
	ID                  string           `json:"id"`
	MemberCount         int              `json:"member_count"`
	CompleteClassCount  int              `json:"complete_class_count"`
	PartialClassCount   int              `json:"partial_class_count"`
	InaccessibleCount   int              `json:"inaccessible_class_count"`
	PartialClasses      []AdjacencyClass `json:"partial_classes"`
	InaccessibleClasses []AdjacencyClass `json:"inaccessible_classes"`
}

type CaptureReceipt struct {
	Repository      string   `json:"repository"`
	Revision        string   `json:"revision"`
	Cutoff          string   `json:"cutoff"`
	Status          string   `json:"status"`
	RecordCount     int      `json:"record_count"`
	CompleteSources []string `json:"complete_sources"`
	IndexChecksum   string   `json:"index_checksum"`
	ReceiptSHA256   string   `json:"receipt_sha256"`
}

type CorpusReceipt struct {
	VLLMRecords          int    `json:"vllm_records"`
	VLLMRevision         string `json:"vllm_revision"`
	VLLMCutoff           string `json:"vllm_cutoff"`
	VLLMIndexChecksum    string `json:"vllm_index_checksum"`
	RelatedRepositories  int    `json:"related_repositories"`
	RelatedForgeComplete int    `json:"related_forge_complete"`
	RelatedRecords       int    `json:"related_records"`
	FAKRecords           int    `json:"fak_records"`
}

type SampleEvidence struct {
	ClusterID      string `json:"cluster_id"`
	Disposition    string `json:"disposition"`
	Actionable     bool   `json:"actionable"`
	ArtifactCount  int    `json:"artifact_count"`
	Confidence     string `json:"confidence"`
	ManualReview   bool   `json:"manual_review"`
	ManualReason   string `json:"manual_reason,omitempty"`
	MembersSHA256  string `json:"members_sha256"`
	EvidenceSHA256 string `json:"evidence_sha256"`
}

type Ledger struct {
	Schema             string           `json:"schema"`
	ParentIssue        int              `json:"parent_issue"`
	Sources            Sources          `json:"sources"`
	Coverage           Coverage         `json:"coverage"`
	Tickets            []Ticket         `json:"tickets"`
	Queue              []QueueEntry     `json:"queue"`
	Adjacency          AdjacencyReceipt `json:"adjacency"`
	Capture            CaptureReceipt   `json:"capture"`
	CorpusReceipt      CorpusReceipt    `json:"corpus_coverage"`
	SamplingEvidence   []SampleEvidence `json:"sampling_evidence"`
	RefreshObligations []string         `json:"refresh_obligations"`
}

type Report struct {
	Schema       string `json:"schema"`
	LedgerSHA256 string `json:"ledger_sha256"`
	Detail       Ledger `json:"-"`
}
