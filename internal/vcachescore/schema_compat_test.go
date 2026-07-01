package vcachescore

import (
	"encoding/json"
	"testing"
)

// legacyReportFields is a fixture of the pre-v1 `fak.vcache.score.v1` JSON
// shape: the fields that existed before the additive `planes`,
// `agentic_activation`, `default_usefulness`, and `cold_path_correct` facets
// were introduced (cache-default[01], #1519). It deliberately omits every
// v1-only key so a round trip through the current Report struct proves the
// old shape still decodes cleanly and a v0-era consumer reading this Report
// would still find every field it used to depend on.
const legacyReportJSON = `{
  "schema": "fak.vcache.score.v1",
  "status": "2x_ready",
  "grade": "A",
  "score": 92,
  "active_source": "telemetry",
  "active_multiplier": 9.6,
  "two_x_threshold": 2,
  "two_x_better": true,
  "planned": {},
  "concentration": {},
  "index": {
    "target_coverage": 0.85,
    "total_anchors": 3,
    "anchor_count": 2,
    "coverage": 0.9
  },
  "prediction": {
    "total": 0,
    "true_warm": 0,
    "false_warm": 0,
    "true_cold": 0,
    "false_cold": 0,
    "false_warm_rate": 0,
    "false_cold_rate": 0
  },
  "recall": {},
  "actions": ["ship it"],
  "risks": []
}`

// TestLegacyReportJSONDecodesWithoutLoss proves the old (pre-facets) score
// JSON shape still unmarshals cleanly into the current Report struct: no
// unknown-field rejection (Report never uses DisallowUnknownFields) and every
// legacy value survives the round trip untouched. This is the "old-format
// parser wouldn't choke" half of the compatibility contract: a consumer that
// only reads the legacy keys sees no data loss when handed a report built by
// the current code.
func TestLegacyReportJSONDecodesWithoutLoss(t *testing.T) {
	var rep Report
	if err := json.Unmarshal([]byte(legacyReportJSON), &rep); err != nil {
		t.Fatalf("legacy report JSON failed to decode into current Report: %v", err)
	}
	if rep.Schema != "fak.vcache.score.v1" {
		t.Fatalf("schema=%q, want fak.vcache.score.v1 preserved", rep.Schema)
	}
	if rep.Status != "2x_ready" || rep.Grade != "A" || rep.Score != 92 {
		t.Fatalf("status/grade/score = %q/%q/%d, want 2x_ready/A/92", rep.Status, rep.Grade, rep.Score)
	}
	if rep.ActiveSource != "telemetry" || rep.ActiveMultiplier != 9.6 {
		t.Fatalf("active_source/active_multiplier = %q/%g, want telemetry/9.6", rep.ActiveSource, rep.ActiveMultiplier)
	}
	if rep.TwoXThreshold != 2 || !rep.TwoXBetter {
		t.Fatalf("two_x_threshold/two_x_better = %g/%v, want 2/true", rep.TwoXThreshold, rep.TwoXBetter)
	}
	if rep.Index.AnchorCount != 2 || rep.Index.Coverage != 0.9 {
		t.Fatalf("index=%+v, want anchor_count=2 coverage=0.9 preserved", rep.Index)
	}
	if len(rep.Actions) != 1 || rep.Actions[0] != "ship it" {
		t.Fatalf("actions=%v, want [\"ship it\"] preserved", rep.Actions)
	}
	// The legacy fixture supplied no v1 facets, so the additive fields must
	// decode to their zero values rather than erroring -- a legacy producer
	// (or hand-built fixture) that never set them must still be a valid
	// Report.
	if rep.Planes != (PlaneReport{}) {
		t.Fatalf("planes=%+v, want zero value when the legacy fixture omitted it", rep.Planes)
	}
	if rep.AgenticActivation != (AgenticActivationReport{}) {
		t.Fatalf("agentic_activation=%+v, want zero value when the legacy fixture omitted it", rep.AgenticActivation)
	}
	if rep.DefaultUsefulness != (DefaultUsefulnessReport{}) {
		t.Fatalf("default_usefulness=%+v, want zero value when the legacy fixture omitted it", rep.DefaultUsefulness)
	}
	if rep.ColdPathCorrect {
		t.Fatalf("cold_path_correct=%v, want false zero value when the legacy fixture omitted it", rep.ColdPathCorrect)
	}
}

// TestCurrentReportJSONKeepsLegacyFieldsAlongsideV1Facets proves the other
// half of the additive-schema contract: marshaling a Report produced by the
// current Score() still emits every legacy top-level key (so an old
// unmarshaler that only reads those keys keeps working) *and* the new
// fak.cache.default_usefulness.v1 facets ride alongside them rather than
// replacing them.
func TestCurrentReportJSONKeepsLegacyFieldsAlongsideV1Facets(t *testing.T) {
	rep := Score(readyScoreInput())

	raw, err := json.Marshal(rep)
	if err != nil {
		t.Fatalf("marshal current Report: %v", err)
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("unmarshal current Report into generic map: %v", err)
	}

	legacyKeys := []string{
		"schema", "status", "grade", "score", "active_source", "active_multiplier",
		"two_x_threshold", "two_x_better", "planned", "concentration", "index",
		"prediction", "recall", "actions", "risks",
	}
	for _, key := range legacyKeys {
		if _, ok := m[key]; !ok {
			t.Fatalf("legacy key %q missing from current Report JSON; additive change must not drop old fields", key)
		}
	}

	v1Keys := []string{"planes", "agentic_activation", "default_usefulness", "cold_path_correct"}
	for _, key := range v1Keys {
		if _, ok := m[key]; !ok {
			t.Fatalf("v1 facet key %q missing from current Report JSON", key)
		}
	}

	var facets struct {
		DefaultUsefulness struct {
			Schema string `json:"schema"`
		} `json:"default_usefulness"`
	}
	if err := json.Unmarshal(raw, &facets); err != nil {
		t.Fatalf("unmarshal default_usefulness facet: %v", err)
	}
	if facets.DefaultUsefulness.Schema != "fak.cache.default_usefulness.v1" {
		t.Fatalf("default_usefulness.schema=%q, want fak.cache.default_usefulness.v1", facets.DefaultUsefulness.Schema)
	}

	// Confirm the legacy schema string is unchanged too: v1 is additive, not
	// a replacement of the top-level score schema.
	if rep.Schema != "fak.vcache.score.v1" {
		t.Fatalf("top-level schema=%q, want fak.vcache.score.v1 unchanged", rep.Schema)
	}
}
