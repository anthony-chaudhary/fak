package quality

import (
	"fmt"
	"strings"
)

// This file is the SET-WIDE COMPARABILITY gate (#5660, under epic #4509): the layer
// that refuses to read a multi-arm campaign as a controlled comparison unless every
// axis the campaign claims to HOLD is witnessed, and identical, across the COMPLETE
// selected arm set.
//
// The defect it exists to catch is specific. A campaign selects three or more arms,
// an operator reads the delta between the two arms they care about — control versus
// the favored treatment — and those two do agree on model, tokenizer, engine, and
// dataset. The comparison looks controlled. But comparability is a property of the
// SET, not of a favored pair: a third arm that quietly ran on a different tokenizer
// is still in the campaign, still contributes to whatever aggregate or ranking the
// campaign reports, and the "held" axis was therefore never held. A pairwise check
// passes and the campaign is confounded anyway. Checking every pair independently
// has the same hole in a different place — it reports which pairs disagree, not that
// the campaign as a whole failed to hold its axes — so this gate evaluates the axis
// over the whole set at once and names the exact partition it found.
//
// Three properties make the gate honest:
//
//   - EVERY selected arm must WITNESS every held axis. An arm that simply does not
//     say which tokenizer it ran under is not evidence that it matched; it is the
//     absence of evidence, and the gate refuses rather than defaulting the axis to
//     "presumably the same as the others".
//   - UNKNOWN dominates DIFFERENT. When any required evidence is missing the verdict
//     is could-not-establish, even if some other axis is already known to differ.
//     The reason is that "these axes differ and no others do" is an ENUMERATION
//     claim, and an enumeration is only sound when every axis was actually seen. A
//     campaign with an unwitnessed axis cannot bound how many ways it is confounded,
//     so it gets the weaker, honest verdict — and the differing axes it DID prove
//     are still reported alongside, so nothing is lost by the demotion.
//   - Every arm is ROLE-TYPED against a closed set. A campaign needs exactly one
//     control for the treatments to be read against; an arm whose role is unstated
//     leaves the shape of the comparison unknown, which is could-not-establish for
//     the same reason an unwitnessed axis is.
//
// The size-two campaign is preserved as the special case it always was, not
// reimplemented: with two arms the set-wide partition of a held axis has more than
// one distinct value exactly when the two arms disagree on it, so the set-wide
// verdict is definitionally the classic pairwise verdict at |arms| = 2. The
// equivalence is asserted against an independent pairwise oracle in the test file
// rather than asserted here in prose.
//
// It is additive in the same sense as post_selection.go and multiple_comparisons.go:
// it registers no oracle, edits no core, and is a pure function of its inputs, so a
// comparability verdict replays. Like the post-selection controller it returns no
// error — every way a campaign can fail to establish comparability rides on the
// decision as a typed, machine-readable finding, because the artifact has to record
// the refusal as faithfully as it records the pass.

// SetComparabilitySchema is the versioned tag on a comparability decision. Consumers
// pin the major so a schema bump is a conscious migration (the #4519 house rule),
// not a silent field drift.
const SetComparabilitySchema = "fak-quality-comparability/1"

// ArmRole is the closed set of roles an arm may play in a campaign. Typing the role
// is what makes the SHAPE of the comparison checkable: a campaign reads its
// treatments against a control, so exactly one control is required and at least one
// treatment must exist for there to be a comparison at all.
type ArmRole string

const (
	// RoleControl is the held-out arm every treatment is read against. Exactly one
	// per campaign — two controls means the campaign never said which baseline its
	// deltas are measured from.
	RoleControl ArmRole = "control"
	// RoleTreatment is an arm that varies whatever the campaign set out to vary. A
	// campaign needs at least one, and every one of them is inside the comparability
	// check — including the ones an operator is not currently looking at.
	RoleTreatment ArmRole = "treatment"
)

// ComparabilityVerdict is the closed set of conclusions the gate may reach. The
// three-way split is load-bearing: collapsing could-not-establish into
// not-comparable would report an absence of evidence as a proven difference, and
// collapsing it into comparable would report it as a pass.
type ComparabilityVerdict string

const (
	// SetComparable: every held axis was witnessed by every selected arm and every
	// arm agreed on it. This is the only verdict that licenses reading the campaign
	// as a controlled comparison.
	SetComparable ComparabilityVerdict = "comparable"
	// SetNotComparable: every required witness was present and at least one held axis
	// takes more than one value across the set. The campaign is PROVABLY confounded,
	// and the finding names which axis and which arms.
	SetNotComparable ComparabilityVerdict = "not_comparable"
	// SetCouldNotEstablish: some required evidence is unknown — an unwitnessed axis,
	// an untyped arm role, or a declaration that names nothing to hold. Never a pass,
	// and never upgraded to a proven difference.
	SetCouldNotEstablish ComparabilityVerdict = "could_not_establish"
)

// ComparabilityFindingCode is the closed vocabulary of typed findings. A structured
// code is what lets a consumer branch on WHY comparability was withheld instead of
// pattern-matching prose.
type ComparabilityFindingCode string

const (
	// FindingDeclarationInvalid: the campaign declares nothing the gate can hold it
	// to — no id, no held axes, or fewer than two selected arms.
	FindingDeclarationInvalid ComparabilityFindingCode = "campaign_declaration_invalid"
	// FindingArmRoleUntyped: an arm names no id, repeats an id already used, or
	// declares a role outside the closed set. The shape of the comparison is then
	// unknown.
	FindingArmRoleUntyped ComparabilityFindingCode = "arm_role_untyped"
	// FindingArmRolesUnbalanced: the typed roles do not form a comparison — not
	// exactly one control, or no treatment at all.
	FindingArmRolesUnbalanced ComparabilityFindingCode = "arm_roles_unbalanced"
	// FindingHeldAxisUnwitnessed: a selected arm hands over no value for a declared
	// held axis. Silence is not agreement.
	FindingHeldAxisUnwitnessed ComparabilityFindingCode = "held_axis_unwitnessed"
	// FindingHeldAxisDiffers: a declared held axis takes more than one value across
	// the selected set. This is the set-wide defect a favored-pair check misses.
	FindingHeldAxisDiffers ComparabilityFindingCode = "held_axis_differs"
)

// blocksEstablishment reports whether a finding leaves the comparability claim merely
// UNKNOWN (as opposed to provably false). Everything except a proven axis
// difference does, which is what makes unknown dominate different.
func (c ComparabilityFindingCode) blocksEstablishment() bool {
	return c != FindingHeldAxisDiffers
}

// CampaignDeclaration is what a multi-arm campaign is held to: which axes it claims
// to hold constant across every selected arm. HeldAxes is the operator's assertion,
// and checking it against what the arms actually witnessed is the whole gate — a
// campaign that declares no held axis is refused rather than passed vacuously,
// because "nothing was held" is not the same claim as "everything matched".
type CampaignDeclaration struct {
	ID       string   `json:"id"`
	HeldAxes []string `json:"held_axes"`
}

// CampaignArm is one SELECTED arm of a campaign: its id, the role it plays, and the
// axis values it witnessed. Axes maps a held-axis name to the value that arm ran
// under; an axis absent from the map (or present but blank) is UNWITNESSED, not
// matched. Every arm the campaign selected belongs here, including the ones outside
// the pair an operator happens to be reading — those are exactly the arms a pairwise
// check drops.
type CampaignArm struct {
	ID   string            `json:"id"`
	Role ArmRole           `json:"role"`
	Axes map[string]string `json:"axes"`
}

// ComparabilityFinding is one typed reason the gate did not certify comparability.
// Axis localizes it to the held axis at fault and Arms names the EXACT arms
// implicated, so a finding is actionable rather than a bare "not comparable".
type ComparabilityFinding struct {
	Code   ComparabilityFindingCode `json:"code"`
	Axis   string                   `json:"axis,omitempty"`
	Arms   []string                 `json:"arms,omitempty"`
	Detail string                   `json:"detail"`
}

// ArmRoleRecord is one arm's adjudicated role: what it declared, and — when the role
// could not be typed — why. It is emitted for every handed arm so a reader can see
// the campaign's shape as the gate understood it.
type ArmRoleRecord struct {
	Index  int     `json:"index"`
	ID     string  `json:"id"`
	Role   ArmRole `json:"role"`
	Typed  bool    `json:"typed"`
	Reason string  `json:"reason,omitempty"`
}

// ComparabilityDecision is the machine-readable output of the gate: the declaration,
// the arm set with every role typed, the verdict, and every finding that produced
// it. Comparable is true only for the SetComparable verdict — a consumer that reads
// nothing but that boolean still fails closed.
type ComparabilityDecision struct {
	Schema   string               `json:"schema"`
	Campaign string               `json:"campaign"`
	Verdict  ComparabilityVerdict `json:"verdict"`
	// Comparable is the single-boolean read of Verdict, false for both refusals.
	Comparable bool     `json:"comparable"`
	HeldAxes   []string `json:"held_axes"`
	// SetSize is the number of arms actually evaluated — the COMPLETE selected set,
	// which is what distinguishes this gate from a favored-pair check.
	SetSize  int                    `json:"set_size"`
	Controls int                    `json:"controls"`
	Arms     []ArmRoleRecord        `json:"arms"`
	Findings []ComparabilityFinding `json:"findings,omitempty"`
	Bound    string                 `json:"bound"`
}

// EvaluateSetComparability adjudicates whether a campaign's complete selected arm
// set is comparable under its declaration. It types every arm's role, requires every
// selected arm to witness every declared held axis, and evaluates each axis over the
// WHOLE set rather than over any one pair.
//
// It returns no error: every failure rides on the decision as a typed finding. It is
// a pure function of (declaration, arms) — arms are visited in the order handed over
// and axes in the order declared, so no map iteration reaches the output and the
// same inputs always produce the same decision, byte for byte.
func EvaluateSetComparability(decl CampaignDeclaration, arms []CampaignArm) ComparabilityDecision {
	d := ComparabilityDecision{
		Schema:   SetComparabilitySchema,
		Campaign: strings.TrimSpace(decl.ID),
		HeldAxes: normalizeHeldAxes(decl.HeldAxes),
		SetSize:  len(arms),
		Arms:     make([]ArmRoleRecord, len(arms)),
	}

	// 1. The declaration. A campaign that names no id, holds no axis, or selects
	// fewer than two arms has not stated a comparison the gate can check.
	if d.Campaign == "" {
		d.find(FindingDeclarationInvalid, "", nil,
			"campaign names no id: a comparability verdict has to be attributable to the campaign it was reached for")
	}
	if len(d.HeldAxes) == 0 {
		d.find(FindingDeclarationInvalid, "", nil,
			"no held axis declared: a campaign that holds nothing fixed has no comparability to establish, and passing it vacuously would certify exactly the confounded comparison this gate exists to catch")
	}
	if len(arms) < 2 {
		d.find(FindingDeclarationInvalid, "", nil, fmt.Sprintf(
			"%d selected arm(s): comparability is a property of a SET, so at least two arms must be handed over", len(arms)))
	}

	// 2. The arms. Role typing runs before the axes because an untyped arm leaves the
	// shape of the comparison unknown, which is its own could-not-establish.
	seen := make(map[string]bool, len(arms))
	controls, treatments, untyped := 0, 0, 0
	for i, a := range arms {
		rec := ArmRoleRecord{Index: i, ID: strings.TrimSpace(a.ID), Role: a.Role, Typed: true}
		switch {
		case rec.ID == "":
			rec.Reason = "arm declares no id: an arm the finding cannot name is an arm an operator cannot go fix"
		case seen[rec.ID]:
			rec.Reason = fmt.Sprintf("duplicate arm id %q: two arms sharing an id cannot be told apart in a finding, and one of them silently shadows the other", rec.ID)
		default:
			switch a.Role {
			case RoleControl:
				controls++
			case RoleTreatment:
				treatments++
			default:
				rec.Reason = fmt.Sprintf("role %q is not one of %q, %q: an arm whose role is unstated leaves the shape of the comparison unknown",
					a.Role, RoleControl, RoleTreatment)
			}
		}
		if rec.Reason != "" {
			rec.Typed, untyped = false, untyped+1
			d.find(FindingArmRoleUntyped, "", []string{armLabel(rec.ID, i)}, fmt.Sprintf("arm %s: %s", armLabel(rec.ID, i), rec.Reason))
		}
		seen[rec.ID] = true
		d.Arms[i] = rec
	}
	d.Controls = controls
	// Role BALANCE is only a meaningful claim once every role is typed: with an
	// untyped arm in the set the counts are partial, and reporting an imbalance
	// derived from partial counts would be a second finding for the same defect.
	if untyped == 0 && len(arms) >= 2 {
		switch {
		case controls != 1:
			d.find(FindingArmRolesUnbalanced, "", nil, fmt.Sprintf(
				"%d arm(s) typed %q: a campaign needs exactly one control for its treatments to be read against", controls, RoleControl))
		case treatments == 0:
			d.find(FindingArmRolesUnbalanced, "", nil, fmt.Sprintf(
				"no arm typed %q: a set of controls is not a comparison", RoleTreatment))
		}
	}

	// 3. The axes, each evaluated over the COMPLETE set. Partitioning the set by the
	// value an arm witnessed is what catches the third arm: the favored pair agreeing
	// only collapses two of the partition's members, it never collapses the third.
	for _, axis := range d.HeldAxes {
		var order []string
		byValue := make(map[string][]string, len(arms))
		var witnessed, unwitnessed []string
		for i, a := range arms {
			label := armLabel(strings.TrimSpace(a.ID), i)
			v := strings.TrimSpace(a.Axes[axis])
			if v == "" {
				unwitnessed = append(unwitnessed, label)
				continue
			}
			if _, ok := byValue[v]; !ok {
				order = append(order, v)
			}
			byValue[v] = append(byValue[v], label)
			witnessed = append(witnessed, label)
		}
		if len(unwitnessed) > 0 {
			d.find(FindingHeldAxisUnwitnessed, axis, unwitnessed, fmt.Sprintf(
				"held axis %q is unwitnessed by %d of %d selected arm(s) (%s): silence is not agreement, so the axis was never established as held",
				axis, len(unwitnessed), len(arms), strings.Join(unwitnessed, ", ")))
		}
		if len(order) > 1 {
			d.find(FindingHeldAxisDiffers, axis, witnessed, fmt.Sprintf(
				"held axis %q takes %d distinct values across the selected set: %s — the axis is not held, so a delta read across these arms is confounded by it",
				axis, len(order), renderAxisPartition(order, byValue)))
		}
	}

	// 4. The verdict. Unknown dominates different: an enumeration of the differing
	// axes is only sound when every axis was actually witnessed.
	d.Verdict = SetComparable
	for _, f := range d.Findings {
		if f.Code.blocksEstablishment() {
			d.Verdict = SetCouldNotEstablish
			break
		}
		d.Verdict = SetNotComparable
	}
	d.Comparable = d.Verdict == SetComparable
	d.Bound = comparabilityBoundStatement(d)
	return d
}

func (d *ComparabilityDecision) find(code ComparabilityFindingCode, axis string, arms []string, detail string) {
	d.Findings = append(d.Findings, ComparabilityFinding{Code: code, Axis: axis, Arms: arms, Detail: detail})
}

// normalizeHeldAxes trims and de-duplicates the declared axes, preserving declared
// order. A repeated axis would otherwise emit the same finding twice for one defect.
func normalizeHeldAxes(axes []string) []string {
	var out []string
	seen := make(map[string]bool, len(axes))
	for _, a := range axes {
		a = strings.TrimSpace(a)
		if a == "" || seen[a] {
			continue
		}
		seen[a] = true
		out = append(out, a)
	}
	return out
}

// armLabel names an arm in a finding, falling back to its position when it declared
// no id — an unnamed arm still has to be locatable in the set that was handed over.
func armLabel(id string, index int) string {
	if id == "" {
		return fmt.Sprintf("arm#%d", index)
	}
	return id
}

// renderAxisPartition renders the value->arms partition of one held axis in the
// order the values were first witnessed, so the differing axis is reported as the
// exact grouping of arms rather than as a bare count.
func renderAxisPartition(order []string, byValue map[string][]string) string {
	parts := make([]string, 0, len(order))
	for _, v := range order {
		parts = append(parts, fmt.Sprintf("%q on [%s]", v, strings.Join(byValue[v], ", ")))
	}
	return strings.Join(parts, "; ")
}

// comparabilityBoundStatement writes down, in words, what the verdict actually
// licenses. It is emitted on the decision so a consumer reads the guarantee off the
// artifact instead of assuming one from the boolean.
func comparabilityBoundStatement(d ComparabilityDecision) string {
	switch d.Verdict {
	case SetComparable:
		return fmt.Sprintf("comparable: all %d selected arm(s) witnessed and agreed on every one of the %d declared held axis/axes, so a delta read across the set is not confounded by them (it says nothing about axes the campaign did not declare)",
			d.SetSize, len(d.HeldAxes))
	case SetNotComparable:
		return fmt.Sprintf("NOT comparable: every required witness was present and %d held axis/axes vary across the %d selected arm(s), so the campaign is provably confounded",
			countFindings(d.Findings, FindingHeldAxisDiffers), d.SetSize)
	default:
		// Count only the findings that actually leave evidence UNKNOWN. A
		// could-not-establish verdict can also carry proven held_axis_differs
		// findings, and counting those here would describe a proven difference as
		// unknown — the exact conflation blocksEstablishment exists to prevent.
		return fmt.Sprintf("could not establish: %d finding(s) leave required evidence unknown across the %d selected arm(s), so comparability is neither proven nor disproven — this is never a pass",
			countBlockingFindings(d.Findings), d.SetSize)
	}
}

// countBlockingFindings counts only the findings that leave the comparability claim
// UNKNOWN. It is deliberately not countFindings(findings, "") — a could-not-establish
// decision can carry proven held_axis_differs findings alongside the unknown ones, and
// folding those into an "unknown" count would report a proven difference as missing
// evidence.
func countBlockingFindings(findings []ComparabilityFinding) int {
	n := 0
	for _, f := range findings {
		if f.Code.blocksEstablishment() {
			n++
		}
	}
	return n
}

// countFindings counts findings of one code, or all of them when code is empty.
func countFindings(findings []ComparabilityFinding, code ComparabilityFindingCode) int {
	n := 0
	for _, f := range findings {
		if code == "" || f.Code == code {
			n++
		}
	}
	return n
}

// ExplainSetComparability renders a ComparabilityDecision as an operator readout. It
// mirrors ExplainPostSelection (post_selection.go): the bridge from a machine verdict
// to "here is exactly which arm broke the comparison". The SET SIZE is always printed
// next to the verdict, because the failure mode this layer exists to catch is an
// operator reading a two-arm conclusion off a campaign that selected more.
func ExplainSetComparability(d ComparabilityDecision) string {
	var b strings.Builder
	campaign := d.Campaign
	if campaign == "" {
		campaign = "(unnamed campaign)"
	}
	switch d.Verdict {
	case SetComparable:
		fmt.Fprintf(&b, "COMPARABLE  %s — %d arm(s), %d held axis/axes established set-wide\n", campaign, d.SetSize, len(d.HeldAxes))
	case SetNotComparable:
		fmt.Fprintf(&b, "NOT COMPARABLE  %s — %d arm(s); a declared held axis varies across the set\n", campaign, d.SetSize)
	default:
		fmt.Fprintf(&b, "COULD NOT ESTABLISH  %s — %d arm(s); required evidence is unknown\n", campaign, d.SetSize)
	}
	fmt.Fprintf(&b, "  held axes: %s\n", joinOrNone(d.HeldAxes))
	for _, a := range d.Arms {
		role := string(a.Role)
		if !a.Typed {
			fmt.Fprintf(&b, "  arm %-24s role %-12s UNTYPED: %s\n", armLabel(a.ID, a.Index), quoteRole(role), a.Reason)
			continue
		}
		fmt.Fprintf(&b, "  arm %-24s role %s\n", armLabel(a.ID, a.Index), role)
	}
	fmt.Fprintf(&b, "  bound: %s\n", d.Bound)
	for i, f := range d.Findings {
		marker := "  "
		if i == 0 {
			marker = "->" // the first thing to fix
		}
		fmt.Fprintf(&b, "%s %s: %s\n", marker, f.Code, f.Detail)
	}
	return b.String()
}

// quoteRole renders an untyped arm's declared role visibly, so an empty role reads
// as an empty role rather than as whitespace in the column.
func quoteRole(role string) string {
	if role == "" {
		return `""`
	}
	return role
}
