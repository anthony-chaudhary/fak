package dispatchtick

import (
	"strings"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// ---------------------------------------------------------------------------
// PER-ISSUE LAUNCH PROFILE — turn an issue's tier metadata into the (model,
// reasoning-effort, ultracode) triple a headless worker is actually launched with,
// so a routine issue runs on the cheap model and an ultra-hard one gets exhaustive
// multi-agent orchestration, instead of every worker taking the seat default.
// ---------------------------------------------------------------------------
//
// This is the launch-side companion to RouteAccountForTier (which picks the SEAT):
// IssueTierFromLabels already parses the tier/T<N>-required|optimal grammar into a
// typed IssueTier; this maps that tier onto one of four canonical launch profiles.
//
// The leaf stays PURE (modelroute + stdlib only): the default table lives here, and
// the cmd/fak shell resolves any operator override from policy and passes a table in,
// exactly as BuildWorkerCommand keeps model/fallback resolution in the shell.
//
// CONSERVATIVE DEGRADE is load-bearing, and it differs from account routing on
// purpose. RouteAccountForTier resolves an UNTAGGED issue to the frontier floor so a
// weak seat is never chosen. Launch profiles do the OPPOSITE for untagged work: they
// report hasProfile=false so the caller keeps today's seat default. Forcing ultracode
// or xhigh onto an unlabeled issue would silently change real token spend, so an
// uplift only ever fires on an explicit tier signal.

// Worker model ids. These are the REAL versioned ids the worker CLI accepts — never
// the bare "fable" alias, which 400-crashes a headless worker (a fleet crash-loop we
// have paid for before). Always the versioned id.
const (
	WorkerModelOpus  = "claude-opus-4-8"
	WorkerModelFable = "claude-fable-5"

	// EffortXHigh is the strongest reasoning-effort knob short of ultracode. It is
	// emitted via the worker CLI's --effort flag (there is no settings.json field).
	EffortXHigh = "xhigh"

	// UltracodeSettingsArg is the Claude --settings payload that puts a worker in
	// ultracode: xhigh per-message reasoning PLUS dynamic multi-agent workflow
	// orchestration. It is session-only (not a persisted settings.json field), so it
	// must be handed per-launch. The interactive launcher's ultracodeSettingsArg
	// (accounts_launch.go) MIRRORS this value by name — the two are kept equal by a
	// drift-guard test rather than an import, matching the by-value mirror idiom
	// tiertag uses for issuecontract's flag vocabulary.
	UltracodeSettingsArg = `{"ultracode":true}`

	// UltraLabel promotes an issue to the ultra-hard bucket (fable + ultracode). The
	// T0/T1/T2 tier vocabulary tops out at T0, so "ultra ultra hard" — the one bucket
	// that runs the cheap model under full multi-agent orchestration — is signalled by
	// this dedicated label. It is self-sufficient: an issue carrying it is uplifted
	// even without co-tagged tier/T0 labels, so the explicit operator intent always wins.
	UltraLabel = "tier/ultra"

	// PMLabel routes project-management coordination work to the cheap model — the
	// launch-side expression of the modelroute pm-fable preset's
	// work_kind=project_management. Triage, next-up ranking, milestone scoring, status
	// rollups, and backlog dedup are bounded, re-runnable coordination (a wrong triage
	// costs a re-run, not a bad production write), so they run on fable at strong
	// reasoning. Like UltraLabel it is self-sufficient — an issue carrying it needs no
	// tier tag — but unlike UltraLabel it YIELDS to an explicit, valid tier tag, so a
	// genuinely hard planning issue additionally tagged tier/T0 still escalates to opus.
	// Matched case-insensitively; inert to the tierLabelRE grammar (never a T<N> tier).
	PMLabel = "tier/pm"
)

// LaunchProfile is the resolved launch configuration for one worker: the model to
// pin, the reasoning-effort knob, and whether to run ultracode (xhigh reasoning PLUS
// dynamic multi-agent workflow orchestration). Effort and Ultracode are mutually
// exclusive on emit — ultracode already implies xhigh — so a canonical profile sets
// exactly one of them.
type LaunchProfile struct {
	Model     string
	Effort    string
	Ultracode bool
}

// The four canonical profiles: {opus, fable} × {xhigh, ultracode}.
var (
	ProfileOpusXHigh      = LaunchProfile{Model: WorkerModelOpus, Effort: EffortXHigh}
	ProfileOpusUltracode  = LaunchProfile{Model: WorkerModelOpus, Ultracode: true}
	ProfileFableXHigh     = LaunchProfile{Model: WorkerModelFable, Effort: EffortXHigh}
	ProfileFableUltracode = LaunchProfile{Model: WorkerModelFable, Ultracode: true}
)

// LaunchBucket names the launch buckets a profile maps from.
type LaunchBucket string

const (
	BucketRoutine LaunchBucket = "routine" // T2 — routine / bounded / low-trust
	BucketNormal  LaunchBucket = "normal"  // T1 — normal implementation
	BucketHard    LaunchBucket = "hard"    // T0 — ultra-hard / high-risk
	BucketUltra   LaunchBucket = "ultra"   // T0 + UltraLabel — the very hardest
	BucketPM      LaunchBucket = "pm"      // PMLabel — project-management coordination
)

// TierLaunchTable maps each bucket to the profile a worker for that bucket launches
// with. A nil or partial table falls back to DefaultTierLaunchTable per bucket.
type TierLaunchTable map[LaunchBucket]LaunchProfile

// DefaultTierLaunchTable encodes the operator guidance: routine work runs on the
// cheap model at strong reasoning; normal/hard escalate the MODEL to opus (hard adds
// ultracode); ultra-hard trades the model back down to fable but turns on ultracode's
// exhaustive multi-agent orchestration — cheap model, maximum orchestration breadth.
// Project-management coordination (BucketPM) runs on the cheap model at strong
// reasoning, exactly like routine work — it is bounded, re-runnable, and low-stakes.
func DefaultTierLaunchTable() TierLaunchTable {
	return TierLaunchTable{
		BucketRoutine: ProfileFableXHigh,
		BucketNormal:  ProfileOpusXHigh,
		BucketHard:    ProfileOpusUltracode,
		BucketUltra:   ProfileFableUltracode,
		BucketPM:      ProfileFableXHigh,
	}
}

// hasLabel reports whether labels contains want (case-insensitive, trimmed), matching
// the lower-cased label grammar tiertag uses. Shared by the self-sufficient launch
// labels (ultra, pm), which select a bucket by exact-label presence rather than tier.
func hasLabel(labels []string, want string) bool {
	for _, l := range labels {
		if strings.EqualFold(strings.TrimSpace(l), want) {
			return true
		}
	}
	return false
}

// hasUltraLabel reports whether the issue carries the explicit ultra promotion label.
func hasUltraLabel(labels []string) bool { return hasLabel(labels, UltraLabel) }

// LaunchBucketForIssue picks the bucket from an issue's labels. The ultra label is a
// self-sufficient, explicit signal and is checked FIRST (the tier vocabulary has no
// level beyond T0). Otherwise the bucket follows the parsed Optimal tier. A
// project-management label is a self-sufficient signal too, but it YIELDS to a valid
// tier: it is consulted only after no trusted tier resolved, so a hard planning issue
// tagged tier/T0 still escalates to opus while a bare PM issue routes to fable. It
// returns ok=false for an untagged or ambiguous issue (no ultra/pm label and
// HasTier=false), so the caller keeps the seat default rather than uplifting work.
func LaunchBucketForIssue(labels []string) (LaunchBucket, bool) {
	if hasUltraLabel(labels) {
		return BucketUltra, true
	}
	it, _ := IssueTierFromLabels(labels)
	if !it.HasTier {
		// No trusted tier signal: a project-management label still routes the issue to
		// its cheap coordination bucket. An explicit VALID tier tag is handled below and
		// wins, so a hard PM planning issue tagged tier/T0 escalates instead.
		if hasLabel(labels, PMLabel) {
			return BucketPM, true
		}
		return "", false
	}
	switch it.Optimal {
	case modelroute.TierT2:
		return BucketRoutine, true
	case modelroute.TierT1:
		return BucketNormal, true
	default: // TierT0 — Optimal is validated >= Required by IssueTierFromLabels
		return BucketHard, true
	}
}

// LaunchProfileForIssue resolves the launch profile for an issue's labels against a
// tier→profile table (nil ⇒ DefaultTierLaunchTable). It returns hasProfile=false for
// an untagged issue, or for a bucket the supplied table leaves undefined, so the
// caller falls back to today's seat-default worker launch. The chosen bucket is
// returned alongside so a status/accounting surface can attribute the decision.
func LaunchProfileForIssue(labels []string, table TierLaunchTable) (profile LaunchProfile, bucket LaunchBucket, hasProfile bool) {
	bucket, ok := LaunchBucketForIssue(labels)
	if !ok {
		return LaunchProfile{}, "", false
	}
	return profileForBucket(bucket, table)
}

// profileForBucket resolves one bucket against an override table, filling any gap (nil
// or partial table) from the built-in default so a defined bucket always resolves.
func profileForBucket(bucket LaunchBucket, table TierLaunchTable) (LaunchProfile, LaunchBucket, bool) {
	if profile, ok := table[bucket]; ok {
		return profile, bucket, true
	}
	if def, ok := DefaultTierLaunchTable()[bucket]; ok {
		return def, bucket, true
	}
	return LaunchProfile{}, bucket, false
}

// Tick-wide work kinds that describe project-management COORDINATION rather than code
// implementation. They mirror dispatchTickOptions.WorkKind / DefaultWorkKind by value
// (not import), the same by-name mirror idiom tiertag uses for issuecontract's grammar.
const (
	WorkKindProjectManagement = "project_management"
	WorkKindGardening         = "gardening"
)

// normalizeWorkKind lower-cases, trims, and folds "-"/" " to "_", so
// "project-management", "Project Management", and "project_management" are one kind.
func normalizeWorkKind(workKind string) string {
	s := strings.ToLower(strings.TrimSpace(workKind))
	s = strings.ReplaceAll(s, "-", "_")
	return strings.ReplaceAll(s, " ", "_")
}

// IsCoordinationWorkKind reports whether a tick-wide work kind is project-management
// coordination (triage, next-up ranking, milestone scoring, status rollups, backlog
// dedup) rather than code implementation. "engineering" — the claude DefaultWorkKind —
// is deliberately NOT coordination, so an ordinary implementation tick is unchanged.
func IsCoordinationWorkKind(workKind string) bool {
	switch normalizeWorkKind(workKind) {
	case WorkKindProjectManagement, WorkKindGardening:
		return true
	}
	return false
}

// LaunchProfileForDispatch resolves a worker's launch profile from BOTH the per-issue
// labels and the tick-wide work kind — the "project management runs on fable BY
// DEFAULT" rung, so a PM dispatch loop needs no per-issue tier/pm label on every issue.
//
// Per-issue labels always WIN: an explicit tier/ultra, a valid tier tag (so hard PM
// planning tagged tier/T0 still escalates to opus), or a bare tier/pm all resolve
// first. Only when the issue carries no trusted label signal does the tick-wide work
// kind decide: a COORDINATION kind routes to the cheap PM bucket, and anything else —
// notably "engineering" — reports hasProfile=false so the caller keeps the seat
// default. The conservative-degrade invariant is preserved: an untagged issue on an
// implementation tick is never uplifted or downgraded.
func LaunchProfileForDispatch(labels []string, workKind string, table TierLaunchTable) (profile LaunchProfile, bucket LaunchBucket, hasProfile bool) {
	if profile, bucket, ok := LaunchProfileForIssue(labels, table); ok {
		return profile, bucket, true
	}
	if !IsCoordinationWorkKind(workKind) {
		return LaunchProfile{}, "", false
	}
	return profileForBucket(BucketPM, table)
}
