package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/mixedprecision"
)

// The tests below are written ADVERSARIALLY against the ticket-13100 INTERFACE_SPEC
// only. None of them may touch non-test source; if the implementation is missing the
// package will fail to compile and that compile error is the honest witness.

// widestFirst is the independent ordering the spec pins (widest -> narrowest). It is
// hardcoded here so a numeric-order regression in the enum cannot hide behind a
// self-referential table.
var widestFirst = []Level{
	LevelF32,
	LevelF16,
	LevelBF16,
	LevelFP8,
	LevelQ8_0,
	LevelQ4_K,
	LevelQ4_0,
	LevelFP4,
	LevelI4,
}

func precisionScheduleTestPoints(layers int) []Point {
	roles := AllRoles()
	points := make([]Point, 0, layers*len(roles)*2)
	for layer := 0; layer < layers; layer++ {
		for _, role := range roles {
			points = append(points, Point{Layer: layer, Role: role, Phase: PhasePrefill})
			points = append(points, Point{Layer: layer, Role: role, Phase: PhaseDecode})
		}
	}
	return points
}

// 1. TOTALITY: UniformSchedule admits every point (layer 0..7, all roles, both phases).
func TestPrecisionScheduleTotality(t *testing.T) {
	points := precisionScheduleTestPoints(8)
	if len(points) < 176 {
		t.Fatalf("point set too small: %d, want >= 176", len(points))
	}
	for _, level := range widestFirst {
		sched := UniformSchedule(level)
		for _, p := range points {
			gotAt := sched.LevelAt(p)
			if gotAt != level {
				t.Fatalf("UniformSchedule(%s).LevelAt(%+v) = %s, want %s", level, p, gotAt, level)
			}
			got, ok := sched.LevelFor(p)
			if !ok {
				t.Fatalf("UniformSchedule(%s).LevelFor(%+v) refused a total schedule point", level, p)
			}
			if got != level {
				t.Fatalf("UniformSchedule(%s).LevelFor(%+v) = %s, want %s", level, p, got, level)
			}
		}
	}
}

// 2. VOCABULARY: String/ParseLevel round-trip, unknown/empty refuse, order preserved.
func TestPrecisionScheduleLevelVocabularyRoundTrip(t *testing.T) {
	tags := []string{"f32", "f16", "bf16", "fp8", "q8_0", "q4_k", "q4_0", "fp4", "i4"}
	if len(tags) != len(widestFirst) {
		t.Fatalf("tag table length %d != level table length %d", len(tags), len(widestFirst))
	}
	for i, level := range widestFirst {
		if uint8(level) != uint8(i) {
			t.Fatalf("level %d numeric value = %d, want %d (widest-first iota order)", i, uint8(level), i)
		}
		if got := level.String(); got != tags[i] {
			t.Fatalf("Level(%d).String() = %q, want %q", level, got, tags[i])
		}
		parsed, err := ParseLevel(level.String())
		if err != nil {
			t.Fatalf("ParseLevel(%q) error: %v", level.String(), err)
		}
		if parsed != level {
			t.Fatalf("ParseLevel(%q) = %s, want %s", level.String(), parsed, level)
		}
	}
	if _, err := ParseLevel(""); err == nil {
		t.Fatalf("ParseLevel(\"\") succeeded, want error")
	}
	if _, err := ParseLevel("nope"); err == nil {
		t.Fatalf("ParseLevel(\"nope\") succeeded, want error")
	}
}

// 3. ROLE: exactly the 11 declared roles are valid; ParseRole refuses unknowns.
func TestPrecisionScheduleRoleValidity(t *testing.T) {
	declared := []Role{
		RoleWeight,
		RoleRoutedExpertUp,
		RoleRoutedExpertDow,
		RoleSharedExpert,
		RoleKVK,
		RoleKVV,
		RoleKVKRaw,
		RoleEngram,
		RoleHead,
		RoleDraft,
		RoleTarget,
	}
	if len(declared) != 11 {
		t.Fatalf("declared role count = %d, want 11", len(declared))
	}
	all := AllRoles()
	if len(all) != len(declared) {
		t.Fatalf("AllRoles() len = %d, want %d", len(all), len(declared))
	}
	seen := map[Role]bool{}
	for _, r := range all {
		seen[r] = true
	}
	for _, r := range declared {
		if !r.Valid() {
			t.Fatalf("Role(%q).Valid() = false, want true", string(r))
		}
		if !seen[r] {
			t.Fatalf("AllRoles() omitted declared role %q", string(r))
		}
	}
	for _, bad := range []Role{"", "Weight", "bogus"} {
		if bad.Valid() {
			t.Fatalf("Role(%q).Valid() = true, want false", string(bad))
		}
	}
	if _, err := ParseRole("bogus"); err == nil {
		t.Fatalf("ParseRole(\"bogus\") succeeded, want error")
	}
	for _, r := range declared {
		got, err := ParseRole(string(r))
		if err != nil || got != r {
			t.Fatalf("ParseRole(%q) = (%q,%v), want (%q,nil)", string(r), got, err, r)
		}
	}
}

// 4. STATIC fail-closed: present point admits, absent point is an explicit (LevelF32,false).
func TestPrecisionScheduleStaticFailClosed(t *testing.T) {
	present := Point{Layer: 1, Role: RoleWeight, Phase: PhasePrefill}
	absent := Point{Layer: 2, Role: RoleHead, Phase: PhaseDecode}
	sched := StaticSchedule(map[Point]Level{present: LevelQ8_0})

	got, ok := sched.LevelFor(present)
	if !ok || got != LevelQ8_0 {
		t.Fatalf("StaticSchedule present LevelFor = (%s,%v), want (%s,true)", got, ok, LevelQ8_0)
	}

	got, ok = sched.LevelFor(absent)
	if ok {
		t.Fatalf("StaticSchedule absent LevelFor(%+v) admitted %s, want refusal", absent, got)
	}
	if got != LevelF32 {
		t.Fatalf("StaticSchedule absent LevelFor level = %s, want %s (fail-closed)", got, LevelF32)
	}
}

// 5. ADAPTER EQUIVALENCE: ScheduleFromAssignments reproduces each assignment in BOTH phases.
func TestPrecisionScheduleFromAssignmentsReproducesAssignments(t *testing.T) {
	moduleRole := map[string]Role{
		"layer.0.mlp":       RoleWeight,
		"layer.1.mlp.gate":  RoleRoutedExpertUp,
		"layer.1.mlp.down":  RoleRoutedExpertDow,
		"layer.0.attn.k":    RoleKVK,
		"layer.0.lm_head":   RoleHead,
		"layer.1.attn.head": RoleTarget,
	}
	roleOf := func(module string) (Role, bool) {
		r, ok := moduleRole[module]
		return r, ok
	}
	descriptor := mixedprecision.Descriptor{
		Schema: mixedprecision.SchemaV1,
		Provenance: mixedprecision.Provenance{
			Artifact: mixedprecision.PinnedRef{ID: "art", Version: "v1", SHA256: hex64},
			Recipe:   mixedprecision.PinnedRef{ID: "recipe", Version: "v1", SHA256: hex64},
			Runtime:  mixedprecision.PinnedRef{ID: "rt", Version: "v1", SHA256: hex64},
		},
		Modules: []mixedprecision.Module{
			{Name: "layer.0.attn.k", Parameters: 1024},
			{Name: "layer.0.lm_head", Parameters: 4096},
			{Name: "layer.0.mlp", Parameters: 8192},
			{Name: "layer.1.attn.head", Parameters: 512},
			{Name: "layer.1.mlp.down", Parameters: 4096},
			{Name: "layer.1.mlp.gate", Parameters: 4096},
		},
		Rules: []mixedprecision.Rule{
			{Pattern: "layer.0.attn.k", Precision: "fp32"},
			{Pattern: "layer.0.lm_head", Precision: "fp16"},
			{Pattern: "layer.0.mlp", Precision: "bf16"},
			{Pattern: "layer.1.attn.head", Precision: "fp8"},
			{Pattern: "layer.1.mlp.gate", Precision: "int8"},
			{Pattern: "layer.1.mlp.down", Precision: "int4"},
		},
		Fallback: mixedprecision.Fallback{Mode: mixedprecision.FallbackRefuse},
	}
	support := mixedprecision.Support{
		Artifacts:  map[string][]string{"art": {"v1"}},
		Recipes:    map[string][]string{"recipe": {"v1"}},
		Runtimes:   map[string][]string{"rt": {"v1"}},
		Precisions: []string{"fp32", "fp16", "bf16", "fp8", "int8", "int4"},
		Combinations: []mixedprecision.Combination{
			{Artifact: "art@v1", Recipe: "recipe@v1", Runtime: "rt@v1", Outcome: mixedprecision.OutcomeSupported},
		},
	}
	result := mixedprecision.Evaluate(descriptor, support)
	if result.Outcome != mixedprecision.OutcomeSupported {
		t.Fatalf("Evaluate outcome = %s (%s/%s), want supported", result.Outcome, result.Reason, result.Detail)
	}
	if len(result.Assignments) == 0 {
		t.Fatalf("Evaluate produced no assignments")
	}

	layerOf := map[string]int{
		"layer.0.attn.k":    0,
		"layer.0.lm_head":   0,
		"layer.0.mlp":       0,
		"layer.1.attn.head": 1,
		"layer.1.mlp.down":  1,
		"layer.1.mlp.gate":  1,
	}
	sched := ScheduleFromAssignments(result.Assignments, roleOf)

	sawLowerPrecision := false
	for _, a := range result.Assignments {
		role, ok := roleOf(a.Module)
		if !ok {
			t.Fatalf("assignment module %q did not map to a role", a.Module)
		}
		want, err := ParseLevel(a.Precision)
		if err != nil {
			t.Fatalf("assignment precision %q for %q did not parse: %v", a.Precision, a.Module, err)
		}
		if want != LevelF32 {
			sawLowerPrecision = true
		}
		layer := layerOf[a.Module]
		for _, phase := range []Phase{PhasePrefill, PhaseDecode} {
			p := Point{Layer: layer, Role: role, Phase: phase}
			got, admitted := sched.LevelFor(p)
			if !admitted {
				t.Fatalf("ScheduleFromAssignments refused %+v (assignment %q -> %s)", p, a.Module, a.Precision)
			}
			if got != want {
				t.Fatalf("ScheduleFromAssignments %+v = %s, want %s (assignment %q)", p, got, want, a.Module)
			}
		}
	}
	if !sawLowerPrecision {
		t.Fatalf("test did not exercise a lower-precision assignment")
	}

	// High-sensitivity modules must have stayed wide: layer 0 attention K is f32.
	kPoint := Point{Layer: 0, Role: RoleKVK, Phase: PhaseDecode}
	if got, ok := sched.LevelFor(kPoint); !ok || got != LevelF32 {
		t.Fatalf("high-sensitivity layer.0.attn.k = (%s,%v), want (%s,true)", got, ok, LevelF32)
	}
}

// 6. EXPLICIT REFUSAL: unknown role and unsupported level refuse with a reason and a receipt entry.
func TestPrecisionScheduleExplicitRefusalTable(t *testing.T) {
	knownPoint := Point{Layer: 0, Role: RoleWeight, Phase: PhaseDecode}
	admitMap := map[Point]Level{knownPoint: LevelQ8_0}
	wideMap := map[Point]Level{knownPoint: Level(200)}
	cases := []struct {
		name  string
		sched Schedule
		point Point
	}{
		{
			name:  "unknown-role",
			sched: StaticSchedule(admitMap),
			point: Point{Layer: 0, Role: Role("bogus"), Phase: PhaseDecode},
		},
		{
			name:  "unsupported-level",
			sched: StaticSchedule(wideMap),
			point: knownPoint,
		},
		{
			// An out-of-enum KV tier must refuse rather than being laundered into F32.
			name:  "unknown-kv-tier",
			sched: ScheduleKVPrecision(compute.KVPrecision(200), LevelF32),
			point: Point{Layer: 0, Role: RoleKVK, Phase: PhaseDecode},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			receipt := &LevelReceipt{}
			level, ok := tc.sched.LevelFor(tc.point)
			if ok {
				t.Fatalf("LevelFor(%+v) admitted %s, want refusal", tc.point, level)
			}
			if level == LevelF32 {
				t.Fatalf("LevelFor(%+v) silently widened to LevelF32, want explicit refusal level", tc.point)
			}
			receipt.Refuse(Resolved{Point: tc.point, Level: level, Reason: "refused by schedule", Source: SourceExplicit})
			if len(receipt.Refused) != 1 {
				t.Fatalf("Refused len = %d, want 1", len(receipt.Refused))
			}
			if receipt.Refused[0].Reason == "" {
				t.Fatalf("Refused[0].Reason is empty, want non-empty")
			}
			if len(receipt.Levels()) != 0 {
				t.Fatalf("refused point was admitted into Levels() = %v", receipt.Levels())
			}
		})
	}
}

// 7. RECEIPT: DistinctLevels counts distinct admitted levels; Levels is first-seen order.
func TestPrecisionScheduleReceiptDistinctLevels(t *testing.T) {
	r := &LevelReceipt{}
	if r.DistinctLevels() != 0 {
		t.Fatalf("zero-value receipt DistinctLevels = %d, want 0", r.DistinctLevels())
	}
	for _, lvl := range []Level{LevelQ8_0, LevelF32, LevelQ8_0, LevelFP8, LevelF32} {
		r.Record(Resolved{Level: lvl, Source: SourceMixedPrecision})
	}
	r.Refuse(Resolved{Level: LevelI4, Reason: "refused", Source: SourceExplicit})
	if got := r.DistinctLevels(); got != 3 {
		t.Fatalf("DistinctLevels = %d, want 3", got)
	}
	want := []Level{LevelQ8_0, LevelF32, LevelFP8}
	got := r.Levels()
	if len(got) != len(want) {
		t.Fatalf("Levels = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("Levels[%d] = %s, want %s (first-seen order)", i, got[i], want[i])
		}
	}
}

// 8. DEFAULT-UNSET UNCHANGED: no schedule behaves byte-identically to today's path.
func TestPrecisionScheduleDefaultUnsetUnchanged(t *testing.T) {
	m := dynamicPrecisionSyntheticModel()
	m.Quantize()
	prompt := []int{3, 17, 5, 23, 41}

	want := m.NewSession()
	want.Quant = true
	wantLogits := want.Prefill(prompt)

	got := m.NewSession()
	got.Quant = true
	if got.PrecisionSchedule != nil {
		t.Fatalf("fresh session PrecisionSchedule = %v, want nil", got.PrecisionSchedule)
	}
	gotLogits := got.Prefill(prompt)
	assertFloat32BitsEqual(t, "default-unset prefill logits", wantLogits, gotLogits)
	assertKVCacheBitsEqual(t, "default-unset prefill", want.Cache, got.Cache)

	id := 11
	wantLogits = want.Step(id)
	gotLogits = got.Step(id)
	assertFloat32BitsEqual(t, "default-unset step logits", wantLogits, gotLogits)
	assertKVCacheBitsEqual(t, "default-unset step", want.Cache, got.Cache)
}

// 9. MULTI-LEVEL FORWARD WITNESS: one pass runs >= 2 distinct levels with a receipt,
// and the argmax agrees with the all-F32 reference.
func TestPrecisionScheduleMultiLevelForwardWitness(t *testing.T) {
	m := dynamicPrecisionSyntheticModel()
	m.Quantize()
	prompt := []int{3, 17, 5, 23, 41}

	reference := m.NewSession()
	reference.Quant = false
	wantLogits := reference.Prefill(prompt)

	got := m.NewSession()
	receipt := &LevelReceipt{}
	got.Quant = true
	got.PrecisionSchedule = StaticSchedule(map[Point]Level{
		{Layer: 0, Role: RoleWeight, Phase: PhasePrefill}: LevelF32,
		{Layer: 1, Role: RoleWeight, Phase: PhasePrefill}: LevelQ8_0,
	})
	got.PrecisionReceipt = receipt

	points := []Point{
		{Layer: 0, Role: RoleWeight, Phase: PhasePrefill},
		{Layer: 1, Role: RoleWeight, Phase: PhasePrefill},
	}
	// The executor runs each point's forward closure at the level the schedule decided,
	// so the levels below are the levels the pass ACTUALLY ran, not a decision side-note.
	ran := map[Level]int{}
	rec, distinct := got.RunScheduledPass(points, func(p Point, level Level) []float32 {
		ran[level]++
		// Execute the real forward arithmetic for this point at its decided level:
		// F32 widens (Quant off), Q8_0 keeps the quantized path.
		wasQuant := got.Quant
		got.Quant = level == LevelQ8_0
		logits := got.Prefill(prompt)
		got.Quant = wasQuant
		return logits
	})
	if rec == nil {
		t.Fatalf("RunScheduledPass returned nil receipt")
	}
	if distinct < 2 || rec.DistinctLevels() < 2 {
		t.Fatalf("distinct levels = %d (receipt %d), want >= 2", distinct, rec.DistinctLevels())
	}
	// Falsification-resistant: every distinct level named by the receipt must be a level
	// the executor actually RAN. Deleting the schedule would make len(ran)==0 and this fail.
	if len(ran) < 2 {
		t.Fatalf("executor ran %d distinct levels, want >= 2 (schedule was not executed)", len(ran))
	}
	wantLevels := map[Level]bool{LevelF32: true, LevelQ8_0: true}
	for lvl, n := range ran {
		if !wantLevels[lvl] {
			t.Fatalf("executor ran unexpected level %s (%d points), want only F32/Q8_0", lvl, n)
		}
		if n == 0 {
			t.Fatalf("level %s recorded with zero executed points", lvl)
		}
	}
	for _, lvl := range rec.Levels() {
		if !wantLevels[lvl] {
			t.Fatalf("receipt named unexpected level %s, want only F32/Q8_0", lvl)
		}
		if _, executed := ran[lvl]; !executed {
			t.Fatalf("receipt named level %s but the executor never ran it", lvl)
		}
	}

	gotLogits := got.Prefill(prompt)
	if len(gotLogits) != len(wantLogits) {
		t.Fatalf("logits len = %d, want %d", len(gotLogits), len(wantLogits))
	}
	if argmax(gotLogits) != argmax(wantLogits) {
		t.Fatalf("argmax = %d, want %d (scheduled pass disagrees with all-F32 reference)",
			argmax(gotLogits), argmax(wantLogits))
	}
	if cos := cosine(gotLogits, wantLogits); cos < 0.99 {
		t.Fatalf("cosine = %.6f, want >= 0.99 vs all-F32 reference", cos)
	}
}

// TestPrecisionScheduleScheduledPassWithoutSinkIsComplete guards the case where the caller
// never installs a Session receipt sink: the RETURNED receipt must still name every admitted
// level the pass spanned. Regression for a pass that executed two distinct levels yet
// returned DistinctLevels()==0 when no sink was set.
func TestPrecisionScheduleScheduledPassWithoutSinkIsComplete(t *testing.T) {
	m := dynamicPrecisionSyntheticModel()
	m.Quantize()

	s := m.NewSession()
	s.Quant = true
	// Deliberately NO s.PrecisionReceipt: the returned receipt is the only record.
	s.PrecisionSchedule = StaticSchedule(map[Point]Level{
		{Layer: 0, Role: RoleWeight, Phase: PhasePrefill}: LevelF32,
		{Layer: 1, Role: RoleWeight, Phase: PhasePrefill}: LevelQ8_0,
	})

	points := []Point{
		{Layer: 0, Role: RoleWeight, Phase: PhasePrefill},
		{Layer: 1, Role: RoleWeight, Phase: PhasePrefill},
	}
	ran := map[Level]int{}
	rec, distinct := s.RunScheduledPass(points, func(p Point, level Level) []float32 {
		ran[level]++
		return nil
	})
	if rec == nil {
		t.Fatalf("RunScheduledPass returned nil receipt with no sink installed")
	}
	if len(ran) != 2 {
		t.Fatalf("executor ran %d distinct levels, want 2", len(ran))
	}
	if distinct != 2 || rec.DistinctLevels() != 2 {
		t.Fatalf("sink-less receipt distinct levels = %d (returned %d), want 2", rec.DistinctLevels(), distinct)
	}
	if len(rec.Levels()) != 2 {
		t.Fatalf("sink-less receipt Levels() = %v, want the two admitted levels", rec.Levels())
	}
}

// --- local helpers (test-only, no source edits) ------------------------------

const hex64 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
