package studyclass

const (
	// FullSchema is the versioned full classification-output contract.
	FullSchema = "fak-studyclass-output/1"
	// CompactSchema is the versioned bounded cluster-index contract.
	CompactSchema = "fak-studyclass-compact-index/1"
	// RulesSchema identifies the deterministic classifier rule set.
	RulesSchema = "fak-studyclass-rules/1"
	// JSONSchemaID is the stable identifier in the embedded JSON Schema document.
	JSONSchemaID = "https://fak.dev/schemas/studyclass-output-1.schema.json"

	DefaultRelatedSampleLimit = 8
)

type Disposition string

const (
	DispositionMergedLanded                Disposition = "merged_landed"
	DispositionOpenProposal                Disposition = "open_proposal"
	DispositionRegressionBug               Disposition = "regression_bug"
	DispositionDuplicate                   Disposition = "duplicate"
	DispositionSupportQuestion             Disposition = "support_question"
	DispositionStaleSuperseded             Disposition = "stale_superseded"
	DispositionClosedUnmerged              Disposition = "closed_unmerged"
	DispositionReleaseMetadataNoncandidate Disposition = "release_metadata_noncandidate"
)

var Dispositions = []Disposition{
	DispositionMergedLanded,
	DispositionOpenProposal,
	DispositionRegressionBug,
	DispositionDuplicate,
	DispositionSupportQuestion,
	DispositionStaleSuperseded,
	DispositionClosedUnmerged,
	DispositionReleaseMetadataNoncandidate,
}

type Mechanism string

const (
	MechanismArchitectureRuntime             Mechanism = "architecture_runtime"
	MechanismSchedulingBatching              Mechanism = "scheduling_batching"
	MechanismKVCache                         Mechanism = "kv_cache"
	MechanismKernelsCompilation              Mechanism = "kernels_compilation"
	MechanismSpeculativeDecoding             Mechanism = "speculative_decoding"
	MechanismDistributedParallelism          Mechanism = "distributed_parallelism"
	MechanismMemoryResidency                 Mechanism = "memory_residency"
	MechanismModelBackendHardware            Mechanism = "model_backend_hardware"
	MechanismAPIsToolCallingStructuredOutput Mechanism = "apis_tool_calling_structured_output"
	MechanismObservabilityOperations         Mechanism = "observability_operations"
	MechanismReliabilitySecurity             Mechanism = "reliability_security"
	MechanismTestsCIDocs                     Mechanism = "tests_ci_docs"
	MechanismExplicitNonCandidate            Mechanism = "explicit_non_candidate"
)

// Mechanisms is the closed issue-taxonomy vocabulary. The taxonomy has eleven
// actionable mechanisms plus its explicit non-candidate member.
var Mechanisms = []Mechanism{
	MechanismArchitectureRuntime,
	MechanismSchedulingBatching,
	MechanismKVCache,
	MechanismKernelsCompilation,
	MechanismSpeculativeDecoding,
	MechanismDistributedParallelism,
	MechanismMemoryResidency,
	MechanismModelBackendHardware,
	MechanismAPIsToolCallingStructuredOutput,
	MechanismObservabilityOperations,
	MechanismReliabilitySecurity,
	MechanismTestsCIDocs,
	MechanismExplicitNonCandidate,
}

type Confidence string

const (
	ConfidenceHigh   Confidence = "high"
	ConfidenceMedium Confidence = "medium"
	ConfidenceLow    Confidence = "low"
)

var Confidences = []Confidence{ConfidenceHigh, ConfidenceMedium, ConfidenceLow}

type InputBinding struct {
	RawSHA256                string `json:"raw_sha256"`
	IndexChecksum            string `json:"index_checksum"`
	Repository               string `json:"repository"`
	Revision                 string `json:"revision"`
	Cutoff                   string `json:"cutoff"`
	CutoffMode               string `json:"cutoff_mode"`
	PostCutoffUpdatedRecords int    `json:"post_cutoff_updated_records"`
	RecordCount              int    `json:"record_count"`
}

// Evidence records only an observed field and the fixed signal a rule matched.
// It deliberately does not infer unavailable GitHub relationships.
type Evidence struct {
	Rule   string `json:"rule"`
	Field  string `json:"field"`
	Signal string `json:"signal"`
}

type Dates struct {
	Created   string `json:"created,omitempty"`
	Updated   string `json:"updated,omitempty"`
	Closed    string `json:"closed,omitempty"`
	Merged    string `json:"merged,omitempty"`
	Published string `json:"published,omitempty"`
}

type MechanismMatch struct {
	Name       Mechanism  `json:"name"`
	Confidence Confidence `json:"confidence"`
	Evidence   []Evidence `json:"evidence"`
}

type Classification struct {
	Identity            string           `json:"identity"`
	Source              string           `json:"source"`
	Kind                string           `json:"kind"`
	ID                  int64            `json:"id"`
	NodeID              string           `json:"node_id,omitempty"`
	Number              int              `json:"number,omitempty"`
	URL                 string           `json:"url,omitempty"`
	Labels              []string         `json:"labels,omitempty"`
	State               string           `json:"state"`
	Dates               Dates            `json:"dates"`
	Merged              bool             `json:"merged,omitempty"`
	Disposition         Disposition      `json:"disposition"`
	Confidence          Confidence       `json:"confidence"`
	DispositionEvidence []Evidence       `json:"disposition_evidence"`
	Mechanisms          []MechanismMatch `json:"mechanisms"`
}

// IdentityRef carries enough upstream fact and rule evidence to audit cluster
// membership without copying title or body text into the output.
type IdentityRef struct {
	Identity    string      `json:"identity"`
	Source      string      `json:"source"`
	Kind        string      `json:"kind"`
	ID          int64       `json:"id"`
	NodeID      string      `json:"node_id,omitempty"`
	Number      int         `json:"number,omitempty"`
	URL         string      `json:"url,omitempty"`
	Labels      []string    `json:"labels,omitempty"`
	State       string      `json:"state"`
	Dates       Dates       `json:"dates"`
	Disposition Disposition `json:"disposition"`
	Confidence  Confidence  `json:"confidence"`
	Evidence    []Evidence  `json:"evidence"`
}

type Cluster struct {
	Key            string        `json:"key"`
	Mechanism      Mechanism     `json:"mechanism"`
	Rule           string        `json:"rule"`
	Signal         string        `json:"signal"`
	Actionable     bool          `json:"actionable"`
	Confidence     Confidence    `json:"confidence"`
	Representative IdentityRef   `json:"representative"`
	Related        []IdentityRef `json:"related"`
}

type Summary struct {
	RecordCount   int            `json:"record_count"`
	ClusterCount  int            `json:"cluster_count"`
	BySource      map[string]int `json:"by_source"`
	ByDisposition map[string]int `json:"by_disposition"`
	ByMechanism   map[string]int `json:"by_mechanism"`
	ByState       map[string]int `json:"by_state"`
	ByConfidence  map[string]int `json:"by_confidence"`
}

type Output struct {
	Schema             string           `json:"schema"`
	Rules              string           `json:"rules"`
	RelationshipPolicy string           `json:"relationship_policy"`
	Input              InputBinding     `json:"input"`
	RecordsChecksum    string           `json:"records_checksum"`
	ClustersChecksum   string           `json:"clusters_checksum"`
	Summary            Summary          `json:"summary"`
	Records            []Classification `json:"records"`
	Clusters           []Cluster        `json:"clusters"`
}

type CompactCluster struct {
	Key             string        `json:"key"`
	Mechanism       Mechanism     `json:"mechanism"`
	Rule            string        `json:"rule"`
	Signal          string        `json:"signal"`
	Actionable      bool          `json:"actionable"`
	Confidence      Confidence    `json:"confidence"`
	MemberCount     int           `json:"member_count"`
	RelatedCount    int           `json:"related_count"`
	MembersChecksum string        `json:"members_checksum"`
	Representative  IdentityRef   `json:"representative"`
	RelatedSamples  []IdentityRef `json:"related_samples"`
}

type CompactIndex struct {
	Schema                  string           `json:"schema"`
	Rules                   string           `json:"rules"`
	RelationshipPolicy      string           `json:"relationship_policy"`
	Input                   InputBinding     `json:"input"`
	FullOutputChecksum      string           `json:"full_output_checksum"`
	RecordsChecksum         string           `json:"records_checksum"`
	ClustersChecksum        string           `json:"clusters_checksum"`
	CompactClustersChecksum string           `json:"compact_clusters_checksum"`
	RelatedSampleLimit      int              `json:"related_sample_limit"`
	Summary                 Summary          `json:"summary"`
	Clusters                []CompactCluster `json:"clusters"`
}
