package model

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/mixedprecision"
)

// precision_schedule.go — the per-point precision PLANNER for a forward pass (#13100).
//
// The kernel already dispatches matmuls on a compute.Dtype, and two adapters already
// decide a precision for a whole span: the dynamic-precision controller
// (dynamic_precision.go, one tier per Prefill/Step) and the mixed-precision contract
// (internal/mixedprecision, metadata only). Neither can answer the question this file
// exists for: WHICH level should THIS tensor run at, at THIS layer, in THIS phase?
//
// So this file introduces a small, auditable schedule layer. A Point names one decision
// site (layer, role, phase); a Schedule answers a Level for it; a LevelReceipt records
// every decision — admitted AND refused — so a pass can be replayed and audited after the
// fact. The ordering is load-bearing (see Level), and the schedules are deliberately
// total-or-fail-closed rather than silently defaulting to the widest level: a missing or
// unsupported mapping is an explicit REFUSAL the caller can see, never a quiet widen that
// would double the byte cost of a tensor nobody asked about.

// Level is a precision level, ordered from WIDEST to NARROWEST.
//
// The ordering is MEANINGFUL, not cosmetic: a LOWER numeric value is a WIDER (more
// precise) level, and vice versa. Any code that "steps down for density" or "would rather
// stay exact" must therefore compare with < / > on Level and not on the raw tag string.
// LevelF32 is 0, so the zero value of Level is the widest possible level — which is why a
// Level returned WITHOUT its accompanying ok=false must never be trusted as a decision:
// the boolean is what distinguishes "chose F32" from "did not choose at all".
type Level uint8

const (
	LevelF32  Level = iota // 0 widest
	LevelF16               // 1
	LevelBF16              // 2
	LevelFP8               // 3
	LevelQ8_0              // 4
	LevelQ4_K              // 5
	LevelQ4_0              // 6
	LevelFP4               // 7
	LevelI4                // 8 narrowest
)

// LevelKVUnknown is the level a schedule reports when it REFUSES a KV-role point because
// the KV tier it was built from is not one it knows. It is deliberately OUTSIDE the ordered
// enum (a value no real level can take) rather than LevelF32: the refusal must be
// distinguishable from "we chose F32", so a caller that inspects the level instead of the
// ok flag cannot misread a refusal as a widest-level widen. It is not parseable and not
// returned by any admitted lookup.
const LevelKVUnknown Level = 255

// levelTags is the canonical tag table, indexed by Level so String and the aliases below
// agree by construction rather than by two parallel switch statements that can drift.
var levelTags = [...]string{
	LevelF32:  "f32",
	LevelF16:  "f16",
	LevelBF16: "bf16",
	LevelFP8:  "fp8",
	LevelQ8_0: "q8_0",
	LevelQ4_K: "q4_k",
	LevelQ4_0: "q4_0",
	LevelFP4:  "fp4",
	LevelI4:   "i4",
}

// levelAliases accepts the mixedprecision vocabulary's own spellings in addition to the
// canonical tags. fp32 is the contract's spelling of the same 32-bit float level as f32;
// int8 is its name for the same level as fp8; int4/nf4 are its names for the same 4-bit
// float level as fp4. They are aliases rather than new Levels because the adapter must
// reproduce an assignment's own precision exactly, and ParseLevel is the one table both the
// adapter and a caller use — a second private mapping would be free to disagree with it.
var levelAliases = map[string]Level{
	"fp32": LevelF32,
	"fp16": LevelF16,
	"int8": LevelFP8,
	"int4": LevelFP4,
	"nf4":  LevelFP4,
}

// String renders the level's canonical dtype tag, e.g. "f32","f16","bf16","fp8",
// "q8_0","q4_k","q4_0","fp4","i4"; unknown -> "level?".
func (l Level) String() string {
	if l == LevelKVUnknown {
		return "kv-unknown"
	}
	if int(l) < len(levelTags) {
		return levelTags[l]
	}
	return "level?"
}

// known reports whether l is inside the declared enum. It is the guard the schedules use
// so an out-of-range Level (a caller-forged Level(200), or a zero value that never came
// from a table) REFUSES instead of rendering as a level.
func (l Level) known() bool { return int(l) < len(levelTags) }

// Dtype maps the level onto the backend dtype the kernels already dispatch on. The boolean
// is false for a Level outside the enum.
//
// Two levels share a backend dtype on purpose: FP4 and I4 are both narrow 4-bit codes and
// the HAL's compute.FP4/I4 distinction (float E2M1 vs integer nibble) is the closest
// existing kernel selector, while Q4_0 has no compute.Dtype of its own today and is
// answered as the generic I4 nibble dtype. These are dispatch HINTS for the existing
// kernels, not a claim that the two levels are numerically identical.
func (l Level) Dtype() (compute.Dtype, bool) {
	switch l {
	case LevelF32:
		return compute.F32, true
	case LevelF16:
		return compute.F16, true
	case LevelBF16:
		return compute.BF16, true
	case LevelFP8:
		return compute.FP8, true
	case LevelQ8_0:
		return compute.Q8_0, true
	case LevelQ4_K:
		return compute.Q4_K, true
	case LevelQ4_0:
		return compute.I4, true
	case LevelFP4:
		return compute.FP4, true
	case LevelI4:
		return compute.I4, true
	default:
		return 0, false
	}
}

// ParseLevel maps a canonical tag (case/space-insensitive) to a Level. An empty string and
// unknown tags are errors so a typo REFUSES rather than silently picking a level — the same
// discipline compute.ParseKVPrecision and mixedprecision's canonicalization use. The
// aliases map above accepts int8/int4/nf4 alongside fp8/fp4.
func ParseLevel(s string) (Level, error) {
	tag := strings.ToLower(strings.TrimSpace(s))
	if tag == "" {
		return LevelF32, fmt.Errorf("model: empty precision level tag")
	}
	for l, t := range levelTags {
		if t == tag {
			return Level(l), nil
		}
	}
	if l, ok := levelAliases[tag]; ok {
		return l, nil
	}
	return LevelF32, fmt.Errorf("model: unknown precision level %q", s)
}

// Role identifies what a tensor is in the pass. It is a typed string so a role parsed from
// a module name and a role written as a literal share one namespace, and so an unknown
// role is distinguishable from an empty one.
type Role string

const (
	RoleWeight          Role = "weight"
	RoleRoutedExpertUp  Role = "routed-expert-up"
	RoleRoutedExpertDow Role = "routed-expert-down"
	RoleSharedExpert    Role = "shared-expert"
	RoleKVK             Role = "kv-k"
	RoleKVV             Role = "kv-v"
	RoleKVKRaw          Role = "kv-k-raw"
	RoleEngram          Role = "engram"
	RoleHead            Role = "head"
	RoleDraft           Role = "draft"
	RoleTarget          Role = "target"
)

// allRoles is the canonical role order AllRoles returns and Role.Valid scans. Declaration
// order is the stable order the spec promises, so adding a role appends here too.
var allRoles = []Role{
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

// Valid reports whether r is one of the declared roles. Comparison is exact: a Role
// literal is a closed vocabulary, so "Weight" (wrong case) is NOT valid — case-folding
// belongs in ParseRole, where a human-supplied token is normalized.
func (r Role) Valid() bool {
	for _, known := range allRoles {
		if r == known {
			return true
		}
	}
	return false
}

// ParseRole maps a role token to a Role; unknown -> error (REFUSES). Unlike ParseLevel it
// case/space-folds first, because role tokens arrive from module names and CLI flags, and
// then returns the canonical literal rather than the caller's spelling.
func ParseRole(s string) (Role, error) {
	tag := Role(strings.ToLower(strings.TrimSpace(s)))
	if tag.Valid() {
		return tag, nil
	}
	return "", fmt.Errorf("model: unknown precision role %q", s)
}

// AllRoles returns every declared role in a stable order. The returned slice is a copy, so
// a caller cannot mutate the package's canonical order.
func AllRoles() []Role {
	out := make([]Role, len(allRoles))
	copy(out, allRoles)
	return out
}

// Phase is the pass phase.
type Phase uint8

const (
	PhasePrefill Phase = iota
	PhaseDecode
)

// String renders the phase as "prefill" | "decode" | "phase?".
func (p Phase) String() string {
	switch p {
	case PhasePrefill:
		return "prefill"
	case PhaseDecode:
		return "decode"
	default:
		return "phase?"
	}
}

// ParsePhase maps a phase token to a Phase; unknown -> error (REFUSES), so a mistyped
// phase cannot silently become the prefill zero value.
func ParsePhase(s string) (Phase, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "prefill":
		return PhasePrefill, nil
	case "decode":
		return PhaseDecode, nil
	default:
		return PhasePrefill, fmt.Errorf("model: unknown precision phase %q", s)
	}
}

// Point is one decision site.
type Point struct {
	Layer int
	Role  Role
	Phase Phase
}

// String renders a point compactly for a receipt line.
func (p Point) String() string {
	return fmt.Sprintf("layer=%d role=%s phase=%s", p.Layer, p.Role, p.Phase)
}

// reasoner is the optional half of the Schedule contract. A schedule that can explain a
// refusal implements it; callers type-assert rather than widening the core two-method
// interface every schedule must carry. The reason is load-bearing for the receipt: a
// refusal recorded with no reason is unauditable.
type reasoner interface {
	refusalReason(p Point) string
}

// Schedule maps a Point to a Level. It is TOTAL over the point domain: a well-formed
// Schedule must return a valid Level for EVERY Point, never a zero value that silently
// means "widest". LevelAt is the raw decision; LevelFor is the checked decision.
type Schedule interface {
	// LevelAt is the raw decision for p.
	LevelAt(p Point) Level
	// LevelFor is the checked decision: it returns the level and whether the schedule
	// ADMITS that level for p (a known role, an in-range level). A refusal is explicit,
	// never a silent widen. On a refusal the returned Level is the level that was
	// REFUSED where one is known (so a caller can see it was e.g. Level(200) or a
	// bogus-role answer), and LevelF32 only for an outright absent point.
	LevelFor(p Point) (Level, bool)
}

// Source names where a level decision came from.
type Source string

const (
	SourceDefault        Source = "default"
	SourceMixedPrecision Source = "mixedprecision"
	SourceKVPrecision    Source = "kvprecision"
	SourceExplicit       Source = "explicit"
)

// Resolved is one recorded level decision.
type Resolved struct {
	Point  Point
	Level  Level
	Reason string
	Source Source
}

// String renders a decision as one receipt line: point, chosen (or refused) level, where
// the decision came from, and why. The reason is included verbatim because a refusal with
// no reason is unauditable — the reason IS the evidence that the refusal was deliberate.
func (r Resolved) String() string {
	src := r.Source
	if src == "" {
		src = SourceDefault
	}
	if r.Reason == "" {
		return fmt.Sprintf("%s level=%s source=%s", r.Point, r.Level, src)
	}
	return fmt.Sprintf("%s level=%s source=%s reason=%s", r.Point, r.Level, src, r.Reason)
}

// LevelReceipt is the auditable trace of a pass. Zero value is usable: both slices are
// nil, Record/Refuse append and allocate, and the read methods return empty results, so a
// caller can declare `var r LevelReceipt` and start recording immediately.
type LevelReceipt struct {
	Resolved []Resolved
	Refused  []Resolved
}

// Record appends an admitted decision.
func (r *LevelReceipt) Record(res Resolved) {
	if r == nil {
		return
	}
	r.Resolved = append(r.Resolved, res)
}

// Refuse appends an explicit refusal. The level carried is the level that was REFUSED (so
// the receipt says what was rejected, not what was used), which is what lets an auditor
// tell "we chose f32" from "we asked for int4 and were told no".
func (r *LevelReceipt) Refuse(res Resolved) {
	if r == nil {
		return
	}
	r.Refused = append(r.Refused, res)
}

// Levels returns the DISTINCT admitted levels in first-seen order. Ordering is
// first-seen (not sorted, not widest-first) because it preserves what the pass actually
// did; a caller that wants a canonical order sorts the result itself.
func (r *LevelReceipt) Levels() []Level {
	if r == nil {
		return nil
	}
	seen := make(map[Level]bool, len(r.Resolved))
	out := make([]Level, 0, len(r.Resolved))
	for _, res := range r.Resolved {
		if !seen[res.Level] {
			seen[res.Level] = true
			out = append(out, res.Level)
		}
	}
	return out
}

// DistinctLevels is the count of distinct admitted levels.
func (r *LevelReceipt) DistinctLevels() int {
	if r == nil {
		return 0
	}
	return len(r.Levels())
}

// --- Concrete schedules ------------------------------------------------------

// uniformSchedule answers one fixed level at every point. It is total by construction:
// there is no point it can fail on, so a valid level admits everywhere.
type uniformSchedule struct{ level Level }

// UniformSchedule returns a total Schedule answering `level` at every Point.
func UniformSchedule(level Level) Schedule { return uniformSchedule{level: level} }

func (s uniformSchedule) LevelAt(Point) Level { return s.level }

func (s uniformSchedule) LevelFor(p Point) (Level, bool) {
	if !s.level.known() {
		return s.level, false
	}
	if !p.Role.Valid() {
		return s.level, false
	}
	return s.level, true
}

// refusalReason names the only two ways a uniform schedule can refuse: an out-of-enum
// level it was constructed with, or a point whose role is not in the vocabulary.
func (s uniformSchedule) refusalReason(p Point) string {
	if !s.level.known() {
		return fmt.Sprintf("model: uniform schedule level %d is out of enum", uint8(s.level))
	}
	return fmt.Sprintf("model: uniform schedule rejects unknown role %q", string(p.Role))
}

// staticSchedule answers only the points it was handed. Every other point is a deliberate
// miss: for an absent point it returns (LevelF32, false) so the caller sees the widest
// level alongside the refusal, and for a present but out-of-enum level it returns that
// refused level, so the receipt names what was actually rejected rather than laundered it
// back to F32.
type staticSchedule struct {
	levels map[Point]Level
	// refusal carries the level a refusal reports for a point that IS covered by an
	// entry but is inadmissible (an unknown role). It is the entry's own level, chosen
	// so a refusal names a real level instead of laundered F32; an ABSENT point still
	// reports LevelF32, since there is no stored level to name.
	refusal Level
}

// StaticSchedule returns a fail-closed Schedule: lookups return the mapped level and true
// only for points present in m with an in-enum level; every other point returns false, and
// a Reason naming the miss via refusalReason.
//
// The map is copied so a caller mutating theirs later cannot change a schedule already in
// flight, which would make a receipt disagree with the pass that produced it.
func StaticSchedule(m map[Point]Level) Schedule {
	levels := make(map[Point]Level, len(m))
	for p, l := range m {
		levels[p] = l
	}
	// A representative refusal level: the first in-enum level in the table, so a bogus
	// role is reported against a level the schedule actually knows rather than F32.
	refusal := LevelF32
	for _, l := range levels {
		if l.known() {
			refusal = l
			break
		}
	}
	return staticSchedule{levels: levels, refusal: refusal}
}

func (s staticSchedule) LevelAt(p Point) Level { return s.levels[p] }

func (s staticSchedule) LevelFor(p Point) (Level, bool) {
	// An unknown role is inadmissible no matter what the table holds, so it refuses
	// against a known level (never a silent F32 widen). Checked before presence so the
	// answer does not depend on which keys happen to be in the map.
	if !p.Role.Valid() {
		return s.refusal, false
	}
	l, ok := s.levels[p]
	if !ok {
		return LevelF32, false
	}
	if !l.known() {
		return l, false
	}
	return l, true
}

// refusalReason explains a fail-closed miss. It distinguishes "not in the table" from
// "in the table but not admissible" so the receipt says which one happened; an empty
// reason would read as "no opinion", whereas this reads as a stated refusal.
func (s staticSchedule) refusalReason(p Point) string {
	l, ok := s.levels[p]
	if !ok {
		return fmt.Sprintf("model: static schedule has no level for %s", p)
	}
	if !p.Role.Valid() {
		return fmt.Sprintf("model: static schedule rejects unknown role %q at %s", string(p.Role), p)
	}
	return fmt.Sprintf("model: static schedule holds out-of-enum level %d for %s", uint8(l), p)
}

// --- mixedprecision adapter --------------------------------------------------

// assignmentSchedule answers a mixedprecision Assignment by (layer, role). It stores the
// assignments' own canonical precision strings as Levels plus a per-point refusal reason,
// so LevelFor can distinguish "no assignment covered this point" from "an assignment
// covered it but names a precision with no Level".
type assignmentSchedule struct {
	levels  map[Point]Level
	reasons map[Point]string
}

// ScheduleFromAssignments builds a total Schedule from resolved mixedprecision Assignment
// values, keyed by (layer, role). The layer is recovered from the assignment's module
// name (layerOfModule), because the contract's Assignment carries no layer field. The
// phase is ignored by this adapter (assignments carry no phase), so its answer is
// phase-invariant: each covered (layer, role) pair is materialized for both phases.
// Canonical precision strings map per mixedprecision's own table: fp32->F32, fp16->F16,
// bf16->BF16, fp8/int8->FP8, int4/nf4/fp4->FP4. int6/int3/int2 have NO Level: those points
// REFUSE explicitly (LevelFor -> false). Unmapped (layer, role) points REFUSE too.
func ScheduleFromAssignments(assignments []mixedprecision.Assignment, roleOf func(module string) (Role, bool)) Schedule {
	levels := make(map[Point]Level, len(assignments))
	reasons := make(map[Point]string, len(assignments))
	if roleOf == nil {
		return assignmentSchedule{levels: levels, reasons: reasons}
	}
	for _, a := range assignments {
		role, ok := roleOf(a.Module)
		if !ok || !role.Valid() {
			continue
		}
		layer, ok := layerOfModule(a.Module)
		if !ok {
			continue
		}
		level, err := ParseLevel(a.Precision)
		mapped := err == nil
		// Assignments carry no phase, so the mapping is written phase-invariant by
		// materializing BOTH phases for the (layer, role) pair. This is what makes the
		// adapter total over the phase axis rather than guessing a phase at lookup time.
		for _, phase := range []Phase{PhasePrefill, PhaseDecode} {
			p := Point{Layer: layer, Role: role, Phase: phase}
			if mapped {
				levels[p] = level
				delete(reasons, p)
				continue
			}
			reasons[p] = fmt.Sprintf(
				"model: mixedprecision precision %q on module %q has no level", a.Precision, a.Module)
		}
	}
	return assignmentSchedule{levels: levels, reasons: reasons}
}

// layerOfModule recovers a layer ordinal from a canonical module name. Module names are
// path-like ("layer.0.mlp", "model.layers.12.attn.k"), so the layer is the integer
// segment that follows a recognised layer marker; failing that, the LAST integer segment,
// which is the convention in every dotted name the contract emits. A name with no integer
// segment at all yields ok=false and its points refuse rather than landing on layer 0 —
// silently stacking every unnamed module onto layer 0 would make two DIFFERENT modules
// collide on one Point and one of them would win by iteration order.
func layerOfModule(module string) (int, bool) {
	parts := strings.FieldsFunc(strings.ToLower(module), func(r rune) bool {
		return r == '.' || r == '/' || r == '_' || r == '-'
	})
	markers := map[string]bool{"layer": true, "layers": true, "blk": true, "block": true, "blocks": true, "h": true}
	best := -1
	for i, part := range parts {
		n, err := strconv.Atoi(part)
		if err != nil {
			continue
		}
		best = n
		if i > 0 && markers[parts[i-1]] {
			return n, true
		}
	}
	if best >= 0 {
		return best, true
	}
	return 0, false
}

func (s assignmentSchedule) LevelAt(p Point) Level { return s.levels[p] }

func (s assignmentSchedule) LevelFor(p Point) (Level, bool) {
	if !p.Role.Valid() {
		return LevelF32, false
	}
	if l, ok := s.levels[p]; ok && l.known() {
		return l, true
	}
	return LevelF32, false
}

// refusalReason is the explicit refusal text for a point an assignment set did not admit.
// It reports WHY (unsupported precision vs uncovered point) so a refusal is actionable
// rather than a bare false.
func (s assignmentSchedule) refusalReason(p Point) string {
	if r, ok := s.reasons[p]; ok {
		return r
	}
	return fmt.Sprintf("model: no mixedprecision assignment covers %s", p)
}

// --- kvprecision adapter -----------------------------------------------------

// kvSchedule is TOTAL: KV-role points answer the tier's level, everything else answers
// the caller's fallback. There is no refusal case for a valid point, so LevelFor is
// always true for one.
type kvSchedule struct {
	kv       compute.KVPrecision
	fallback Level
}

// ScheduleKVPrecision returns a TOTAL Schedule whose KV-role points (RoleKVK, RoleKVV,
// RoleKVKRaw) answer the level named by kv, and whose non-KV points answer fallback.
// KVPrecisionF32 -> LevelF32; KVPrecisionQ8 -> LevelQ8_0.
//
// The KV roles are exactly the three rows the cache holds (pre-RoPE K, post-RoPE K, V),
// which is why the tier maps onto all three rather than only the attended two: the planner
// needs a level for every row it will write, and compute.KVPrecision already encodes the
// mixed layout internally.
func ScheduleKVPrecision(kv compute.KVPrecision, fallback Level) Schedule {
	return kvSchedule{kv: kv, fallback: fallback}
}

// kvLevel maps a KV tier onto the level of the three cache rows. The boolean is false for a
// tier this schedule does not know, so an out-of-enum compute.KVPrecision REFUSES instead of
// falling through to F32 — an unknown tier must never be laundered into the widest level.
//
// On a refusal the returned level is LevelKVUnknown rather than LevelF32: the whole point of
// the refusal is that no real level was chosen, and returning F32 would let a caller that
// ignores the boolean run the widest level under a "widened" story. LevelAt has no better
// answer to give and also reports LevelKVUnknown for an unknown tier.
func kvLevel(kv compute.KVPrecision) (Level, bool) {
	switch kv {
	case compute.KVPrecisionF32:
		return LevelF32, true
	case compute.KVPrecisionQ8:
		return LevelQ8_0, true
	default:
		return LevelKVUnknown, false
	}
}

func (s kvSchedule) LevelAt(p Point) Level {
	if isKVRole(p.Role) {
		if l, ok := kvLevel(s.kv); ok {
			return l
		}
		return LevelKVUnknown
	}
	return s.fallback
}

func (s kvSchedule) LevelFor(p Point) (Level, bool) {
	if !p.Role.Valid() {
		return s.LevelAt(p), false
	}
	if isKVRole(p.Role) {
		// An unknown KV tier refuses even though LevelAt has no better answer to give.
		return kvLevel(s.kv)
	}
	l := s.fallback
	if !l.known() {
		return l, false
	}
	return l, true
}

// refusalReason is non-empty for the only refusals this schedule has: an unknown role, an
// out-of-enum KV tier, or an out-of-enum fallback level. A valid KV or valid fallback point
// never refuses.
func (s kvSchedule) refusalReason(p Point) string {
	if !p.Role.Valid() {
		return fmt.Sprintf("model: kvprecision schedule rejects unknown role %q", string(p.Role))
	}
	if isKVRole(p.Role) {
		if _, ok := kvLevel(s.kv); !ok {
			return fmt.Sprintf("model: kvprecision schedule rejects unknown KV tier %d at %s", uint8(s.kv), p)
		}
	}
	return fmt.Sprintf("model: kvprecision schedule fallback level %d is out of enum at %s", uint8(s.fallback), p)
}

// isKVRole reports whether a role is one of the three cache rows the KV tier governs.
func isKVRole(r Role) bool {
	switch r {
	case RoleKVK, RoleKVV, RoleKVKRaw:
		return true
	default:
		return false
	}
}

// scheduleReason renders the refusal reason a schedule can explain, or "" when the
// schedule implements no explainer. scheduledLevel uses it to populate the receipt, so
// every refusal a pass records carries a non-empty reason.
func scheduleReason(s Schedule, p Point) string {
	if r, ok := s.(reasoner); ok {
		return r.refusalReason(p)
	}
	return ""
}
