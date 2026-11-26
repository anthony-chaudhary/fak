package scorecard

// The payload types mirror the Python payload's key order field by field, so
// encoding/json emits the keys in the order json.dumps does.

// KPI is one per-KPI check.
type KPI struct {
	Kpi     string   `json:"kpi"`
	Group   string   `json:"group"`
	Score   int      `json:"score"`
	Value   PyFloat  `json:"value"`
	Detail  string   `json:"detail"`
	Defects []string `json:"defects"`
	Soft    []string `json:"soft"`
}

// Breakdown is one row of the worst-first KPI table.
type Breakdown struct {
	Kpi    string  `json:"kpi"`
	Group  string  `json:"group"`
	Score  int     `json:"score"`
	Value  PyFloat `json:"value"`
	Debt   int     `json:"debt"`
	Detail string  `json:"detail"`
}

// FamilyCoverage is one family's discovered/covered view.
type FamilyCoverage struct {
	Family     Value    `json:"family"`
	Discovered int      `json:"discovered"`
	Covered    int      `json:"covered"`
	Uncovered  []string `json:"uncovered"`
}

// UncoveredToken is one unpositioned tree token and the family that found it.
type UncoveredToken struct {
	Family Value  `json:"family"`
	Token  string `json:"token"`
}

// Coverage is the coverage report.
type Coverage struct {
	Discovered   int              `json:"discovered"`
	Covered      int              `json:"covered"`
	CoveragePct  PyFloat          `json:"coverage_pct"`
	CoverageDebt int              `json:"coverage_debt"`
	PerFamily    []FamilyCoverage `json:"per_family"`
	Uncovered    []UncoveredToken `json:"uncovered"`
}

// LeaderRow is one concept in the leaderboard.
type LeaderRow struct {
	ID              Value  `json:"id"`
	Canonical       Value  `json:"canonical"`
	Family          Value  `json:"family"`
	Kind            Value  `json:"kind"`
	Verdict         Value  `json:"verdict"`
	ExpectedVerdict string `json:"expected_verdict"`
	Definition      Value  `json:"definition"`
	DistinctFrom    Value  `json:"distinct_from"`
	Grounding       Value  `json:"grounding"`
	GlossaryAnchor  Value  `json:"glossary_anchor"`
}

// CriticalRow is one entry of the worst-first clarity backlog.
type CriticalRow struct {
	ID        Value `json:"id"`
	Canonical Value `json:"canonical"`
	Family    Value `json:"family"`
	Verdict   Value `json:"verdict"`
	Debt      int   `json:"debt"`
	Distance  int   `json:"distance"`
	Gaps      Value `json:"gaps"`
}

// Weakest names the foggiest concept of a subtree.
type Weakest struct {
	ID      string `json:"id"`
	Verdict Value  `json:"verdict"`
}

// Abstraction is one abstraction head rolled up weakest-link.
type Abstraction struct {
	ID              string        `json:"id"`
	Canonical       Value         `json:"canonical"`
	Family          Value         `json:"family"`
	DeclaredVerdict Value         `json:"declared_verdict"`
	RolledVerdict   string        `json:"rolled_verdict"`
	SubtreeSize     int           `json:"subtree_size"`
	ChildCount      int           `json:"child_count"`
	SubtreeOwed     int           `json:"subtree_debt"`
	VerdictMix      OrderedCounts `json:"verdict_mix"`
	Depth           int           `json:"depth"`
	Weakest         Weakest       `json:"weakest"`
	Overclaim       bool          `json:"overclaim"`
}

// Rollup is the hierarchy roll-up.
type Rollup struct {
	Abstractions []Abstraction `json:"abstractions"`
	Heads        int           `json:"heads"`
	Roots        int           `json:"roots"`
	ForestNodes  int           `json:"forest_nodes"`
	MaxDepth     int           `json:"max_depth"`
	Overclaims   []string      `json:"overclaims"`
}

// GradedPair is a confusable pair graded against the drawn boundaries.
type GradedPair struct {
	A          string `json:"a"`
	B          string `json:"b"`
	Kind       string `json:"kind"`
	Why        string `json:"why"`
	State      string `json:"state"`
	AToB       bool   `json:"a_to_b"`
	BToA       bool   `json:"b_to_a"`
	ACanonical Value  `json:"a_canonical"`
	BCanonical Value  `json:"b_canonical"`
	AFamily    Value  `json:"a_family"`
	BFamily    Value  `json:"b_family"`
}

// Unseparated is a confusable pair whose boundary is not drawn both ways.
type Unseparated struct {
	A     string `json:"a"`
	B     string `json:"b"`
	Kind  string `json:"kind"`
	Why   string `json:"why"`
	State string `json:"state"`
	AToB  bool   `json:"a_to_b"`
	BToA  bool   `json:"b_to_a"`
}

// Separation is the payload's pairwise-separation section.
type Separation struct {
	ConfusablePairs    int           `json:"confusable_pairs"`
	Separated          int           `json:"separated"`
	Mutual             int           `json:"mutual"`
	OneSided           int           `json:"one_sided"`
	Undrawn            int           `json:"undrawn"`
	EntangledConcepts  int           `json:"entangled_concepts"`
	Boundaries         int           `json:"boundaries"`
	MutualBoundaries   int           `json:"mutual_boundaries"`
	OneWayBoundaries   int           `json:"one_way_boundaries"`
	DanglingReferences int           `json:"dangling_references"`
	Pairs              []GradedPair  `json:"pairs"`
	Unseparated        []Unseparated `json:"unseparated"`
}

// IndexEntry is one lookup key of the name index.
type IndexEntry struct {
	Key       string         `json:"key"`
	Spellings []string       `json:"spellings"`
	Targets   []string       `json:"targets"`
	Via       OrderedSources `json:"via"`
	Ambiguous bool           `json:"ambiguous"`
}

// IndexSummary is the payload's lean index section.
type IndexSummary struct {
	Keys                  int          `json:"keys"`
	AmbiguousKeys         int          `json:"ambiguous_keys"`
	SharedNamePairs       int          `json:"shared_name_pairs"`
	UnresolvedSharedNames int          `json:"unresolved_shared_names"`
	Ambiguous             []IndexEntry `json:"ambiguous"`
	ContrastConcepts      int          `json:"contrast_concepts"`
}

// Index is the full name index that rides outside corpus.
type Index struct {
	Entries       []IndexEntry      `json:"entries"`
	Keys          int               `json:"keys"`
	Ambiguous     []IndexEntry      `json:"ambiguous"`
	AmbiguousKeys int               `json:"ambiguous_keys"`
	ByKey         OrderedKeyTargets `json:"by_key"`
}

// Corpus is the payload's corpus section.
type Corpus struct {
	Value              PyFloat       `json:"value"`
	ValueUnit          string        `json:"value_unit"`
	Score              PyFloat       `json:"score"`
	LegacyScore        PyFloat       `json:"legacy_score"`
	LegacyScoreScale   int           `json:"legacy_score_scale"`
	Grade              string        `json:"grade"`
	ClarityScore       PyFloat       `json:"clarity_score"`
	ClarityValue       PyFloat       `json:"clarity_value"`
	DisambiguationDebt int           `json:"disambiguation_debt"`
	ClarityDefects     int           `json:"clarity_defects"`
	CoverageDebt       int           `json:"coverage_debt"`
	Coverage           Coverage      `json:"coverage"`
	SoftSignals        int           `json:"soft_signals"`
	Rows               int           `json:"rows"`
	CrystalConcepts    int           `json:"crystal_concepts"`
	Families           int           `json:"families"`
	AsOf               Value         `json:"as_of"`
	FakVersion         Value         `json:"fak_version"`
	Standing           OrderedCounts `json:"standing"`
	DebtByGroup        OrderedCounts `json:"debt_by_group"`
	KpiScores          OrderedCounts `json:"kpi_scores"`
	DebtByKpi          OrderedCounts `json:"debt_by_kpi"`
	Breakdown          []Breakdown   `json:"breakdown"`
	Leaderboard        []LeaderRow   `json:"leaderboard"`
	Critical           []CriticalRow `json:"critical"`
	Rollup             Rollup        `json:"rollup"`
	Separation         Separation    `json:"separation"`
	Index              IndexSummary  `json:"index"`
}

// DataEcho is the `_data` echo of the rows and family definitions folded.
type DataEcho struct {
	Rows     []*Object `json:"rows"`
	Families []*Object `json:"families"`
}

// Payload is one scorecard fold. Corpus is nil for an AUDIT_ERROR payload.
type Payload struct {
	Schema     string    `json:"schema"`
	OK         bool      `json:"ok"`
	Verdict    string    `json:"verdict"`
	Finding    string    `json:"finding"`
	Reason     string    `json:"reason"`
	NextAction string    `json:"next_action"`
	Workspace  string    `json:"workspace"`
	Corpus     *Corpus   `json:"corpus"`
	Kpis       []KPI     `json:"kpis"`
	Data       *DataEcho `json:"_data"`
	Index      *Index    `json:"_index"`
}
