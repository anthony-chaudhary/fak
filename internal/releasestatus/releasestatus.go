// Package releasestatus folds the full read-only release posture that
// tools/release_status.py emits into one typed Go record — the broader sibling
// of internal/releasestale, which only covers the publish-staleness slice.
//
// release_status.py is the operator front door for the release process: it folds
// the rolling release decision, optional cut dry-run, cadence-workflow presence,
// GitHub release-page state, dirty release-relevant path detection, stable-channel
// lag + evidence checks, and a next-action classification into one JSON record. It
// never edits files, commits, tags, pushes, or writes scratch. This package ports
// that contract WITHOUT silently narrowing it: every evidence sub-record the python
// emits has a typed home here, and the classification rules (dirty-relevance,
// stable evidence, next-action, loop-status) are pure functions of injected Facts
// so they are deterministic and unit-testable without git, gh, or python.
//
// The split mirrors releasestale: Compute over Facts. Facts is the injected
// git/external evidence (the rolling decision payload, the stable tags + their
// evidence frontmatter, the dirty porcelain, the cadence yaml text, the gh release
// view); Fold is the pure folder that turns Facts into a Status with the same
// schema/next_action/ok/verdict envelope the python record carries. Gathering the
// Facts via git/gh/python is the impure shell the cmd layer owns; this package does
// the deterministic part the issue centers on — the fold, not the I/O.
//
// It performs NO release: it is pure observation, exactly like the python it ports.
package releasestatus

import (
	"fmt"
	"sort"
	"strings"
)

// Schema is the stable envelope id, matching tools/release_status.py's SCHEMA so a
// consumer of the native fold sees the same record id as the python emitted.
const Schema = "fleet-release-status/1"

// actionNextActions is the closed set of next_action.kind values that mean a human
// or loop must act (loop-status verdict ACTION). Any kind outside it reads OK. This
// is the Go port of release_status.py's ACTION_NEXT_ACTIONS plus the two
// non-gating advisory kinds the python also emits (wait, pause_auto_release).
var actionNextActions = map[string]bool{
	"clean_worktree":         true,
	"cut_release":            true,
	"fix_ci_billing":         true,
	"fix_ci":                 true,
	"confirm_ci":             true,
	"fix_workflow":           true,
	"fix_version_topology":   true,
	"repair_stable_evidence": true,
	"promote_stable":         true,
	"cut_release_hot_tree":   true,
	"hold":                   true,
	"consider_stable":        true,
}

// releaseRelevantExact / releaseRelevantPrefixes mirror release_status.py's
// RELEASE_RELEVANT_DIRTY_EXACT / RELEASE_RELEVANT_DIRTY_PREFIXES: the working-tree
// paths whose dirtiness blocks treating release status as trunk evidence.
var releaseRelevantExact = map[string]bool{
	"VERSION":                    true,
	".claude/project.yaml":       true,
	"tools/safe_ff_sync.py":      true,
	"tools/safe_ff_sync_test.py": true,
}

var releaseRelevantPrefixes = []string{
	".claude/skills/release/",
	".claude/skills/stable-release/",
	".github/workflows/",
	"docs/releases/",
	"docs/stable-releases/",
	"tools/release_",
	"tools/stable_release_",
}

// ---- injected facts -----------------------------------------------------------

// Decision is the rolling release_decide.py verdict slice the fold consumes. Only
// the fields the fold reads are typed; the cmd layer carries the full payload for
// the JSON surface.
type Decision struct {
	Decision    string   `json:"decision"`     // "release" | "hold" | "unknown" | ...
	Reason      string   `json:"reason"`       // human reason line
	NextVersion string   `json:"next_version"` // the version a cut would mint
	Blockers    []string `json:"blockers"`     // structured blocker codes (CI_BASE_RED, ...)
}

// DirtyEntry is one porcelain row: a path and whether git reported it untracked.
type DirtyEntry struct {
	Path      string
	Untracked bool
}

// StableTag is one refs/tags/stable/* row with its resolved sha and VERSION.
type StableTag struct {
	Tag       string `json:"tag"`
	SHA       string `json:"sha"`
	ShortSHA  string `json:"short_sha"`
	CreatedAt string `json:"created_at"`
	Version   string `json:"version"`
	// Evidence is the parsed frontmatter of docs/stable-releases/<codename>.md at
	// HEAD ("" key missing == file absent). EvidenceFound distinguishes an absent
	// file from one with empty/unparseable frontmatter.
	Evidence      map[string]string
	EvidenceFound bool
	// UnderlyingTagSHA is the sha the frontmatter.underlying_version rolling tag
	// resolves to (for the evidence cross-check); "" if that tag does not exist.
	UnderlyingTagSHA string
	UnderlyingExists bool
}

// CIDiagnosis is the ci.yml failure classification slice the fold consumes for the
// CI_BASE_RED next-action branch (kind=="fix_ci_billing" when billing was detected).
type CIDiagnosis struct {
	Action string `json:"action"`
	Detail string `json:"detail"`
}

// CadenceText is the release-cadence.yml posture: whether the file is present and
// its raw text, so the fold can derive the same posture booleans the python does.
type CadenceText struct {
	Present bool
	Path    string
	Text    string
}

// Facts is the injected evidence the fold computes over — everything
// release_status.py gathers via git/gh/python, made explicit so Fold is pure.
type Facts struct {
	Root                 string
	HeadSHA              string
	Branch               string
	Decision             Decision
	RollingTags          []string // semver tags merged into HEAD, ascending (semver_tags(merged=True))
	LastTag              string   // context.last_tag, else the newest rolling tag
	CommitsSinceTag      int
	FilesTouchedSinceTag int
	Dirty                []DirtyEntry
	CIDiagnosis          CIDiagnosis
	Cadence              CadenceText
	Stable               []StableTag // refs/tags/stable/* newest-first (for-each-ref --sort=-creatordate)
	// StableWindowDays is the soak age a rolling candidate must reach before stable
	// promotion is "ready" (release_status.py --stable-window-days, default 3).
	StableWindowDays float64
	// CandidateAgeDays is the age in days of the newest rolling tag (rolling[-1]).
	// 0 means unknown age (its kind reads unknown_age, not soaking).
	CandidateAgeDays  float64
	CandidateAgeKnown bool
	// SuggestedCodename is the next free YYYY-MM-stable[-N] codename, gathered by the
	// impure shell (it needs git to test tag/evidence-file existence).
	SuggestedCodename string
	// CandidateSHA is the commit the newest rolling tag (rolling[-1]) resolves to,
	// gathered by the shell via `git rev-list -n1 <tag>`. "" means git could not
	// resolve it, which the fold reports as candidate state "unresolved" (mirroring
	// the python tag_sha() returning None).
	CandidateSHA string
}

// ---- the typed status record --------------------------------------------------

// DirtySummary mirrors release_status.py dirty_summary().
type DirtySummary struct {
	Clean                bool     `json:"clean"`
	ModifiedCount        int      `json:"modified_count"`
	UntrackedCount       int      `json:"untracked_count"`
	Modified             []string `json:"modified"`
	Untracked            []string `json:"untracked"`
	ReleaseRelevantCount int      `json:"release_relevant_count"`
	ReleaseRelevant      []string `json:"release_relevant"`
	UnrelatedCount       int      `json:"unrelated_count"`
	Unrelated            []string `json:"unrelated"`
}

// Cadence mirrors release_status.py cadence_status().
type Cadence struct {
	Present              bool   `json:"present"`
	Path                 string `json:"path"`
	Schedule             bool   `json:"schedule,omitempty"`
	ManualDispatch       bool   `json:"manual_dispatch,omitempty"`
	DryRunFirst          bool   `json:"dry_run_first,omitempty"`
	SingleWriter         bool   `json:"single_writer,omitempty"`
	TagAfterGreen        bool   `json:"tag_after_green,omitempty"`
	CheckedGitHubRelease bool   `json:"checked_github_release,omitempty"`
}

// StableEvidenceRow mirrors one row of release_status.py stable_evidence_status().
type StableEvidenceRow struct {
	Tag          string   `json:"tag"`
	EvidencePath string   `json:"evidence_path"`
	OK           bool     `json:"ok"`
	Issues       []string `json:"issues"`
}

// StableEvidenceFailure is one flattened (tag, path, detail) failure.
type StableEvidenceFailure struct {
	Tag          string `json:"tag"`
	EvidencePath string `json:"evidence_path"`
	Detail       string `json:"detail"`
}

// StableEvidence mirrors release_status.py stable_evidence_status().
type StableEvidence struct {
	OK       bool                    `json:"ok"`
	Checked  int                     `json:"checked"`
	Failures []StableEvidenceFailure `json:"failures"`
	Rows     []StableEvidenceRow     `json:"rows"`
}

// StableCandidate mirrors release_status.py stable_candidate_status().
type StableCandidate struct {
	CandidateTag      string  `json:"candidate_tag,omitempty"`
	CandidateSHA      string  `json:"candidate_sha,omitempty"`
	WindowDays        float64 `json:"window_days"`
	Ready             bool    `json:"ready"`
	State             string  `json:"state"`
	Reason            string  `json:"reason"`
	StableTag         string  `json:"stable_tag,omitempty"`
	AgeDays           float64 `json:"age_days,omitempty"`
	RemainingDays     float64 `json:"remaining_days,omitempty"`
	SuggestedCodename string  `json:"suggested_codename,omitempty"`
}

// StableSummary mirrors release_status.py stable_summary().
type StableSummary struct {
	LatestStable     *StableTag      `json:"latest_stable"`
	StableLag        *int            `json:"stable_lag"`
	NewerRollingTags []string        `json:"newer_rolling_tags,omitempty"`
	Evidence         StableEvidence  `json:"evidence"`
	Candidate        StableCandidate `json:"candidate"`
	Recommendation   string          `json:"recommendation"`
}

// NextAction mirrors release_status.py next_action(): the single operator move.
type NextAction struct {
	Kind   string `json:"kind"`
	Detail string `json:"detail"`
}

// BranchRegimeFacts is the injected evidence for the dev/main promotion-status slice.
// It is intentionally separate from the rolling/stable release fold: the branch-regime
// view answers whether development has drifted from the release front door, while the
// rolling fold answers whether the latest tag/release is stale.
type BranchRegimeFacts struct {
	DevelopmentBranch string
	DevelopmentHead   string
	ReleaseBranch     string
	ReleaseHead       string
	ReleaseSource     string
	PublicFrontDoor   string
	LatestTag         string
	DevelopmentAhead  int
	ReleaseAhead      int
	DevelopmentCI     string
	ReleaseLockHeld   bool
	RoleError         string
}

// BranchRegime is the machine-readable dev/main drift and promotion-blocker summary.
type BranchRegime struct {
	DevelopmentBranch  string   `json:"development_branch"`
	DevelopmentHead    string   `json:"development_head,omitempty"`
	ReleaseBranch      string   `json:"release_branch"`
	ReleaseHead        string   `json:"release_head,omitempty"`
	ReleaseSource      string   `json:"release_source,omitempty"`
	PublicFrontDoor    string   `json:"public_front_door,omitempty"`
	LatestTag          string   `json:"latest_tag,omitempty"`
	DevelopmentAhead   int      `json:"development_ahead"`
	ReleaseAhead       int      `json:"release_ahead"`
	Drift              string   `json:"drift"`
	DevelopmentCI      string   `json:"development_ci,omitempty"`
	PromotionCandidate string   `json:"promotion_candidate,omitempty"`
	PromotionBlocked   bool     `json:"promotion_blocked"`
	PromotionBlockers  []string `json:"promotion_blockers"`
	ReleaseLockHeld    bool     `json:"release_lock_held"`
	NextAction         string   `json:"next_action"`
	RoleError          string   `json:"role_error,omitempty"`
}

// Status is the full folded record — the typed Go port of the python JSON.
type Status struct {
	Schema               string        `json:"schema"`
	Root                 string        `json:"root"`
	HeadSHA              string        `json:"head_sha"`
	Branch               string        `json:"branch"`
	Dirty                DirtySummary  `json:"dirty"`
	LastTag              string        `json:"last_tag"`
	CommitsSinceTag      int           `json:"commits_since_tag"`
	FilesTouchedSinceTag int           `json:"files_touched_since_tag"`
	Decision             string        `json:"decision"`        // the rolling release_decide verb
	DecisionReason       string        `json:"decision_reason"` // its reason line
	Cadence              Cadence       `json:"cadence"`
	Stable               StableSummary `json:"stable"`
	NextAction           NextAction    `json:"next_action"`
	OK                   bool          `json:"ok"`
	Verdict              string        `json:"verdict"`
	Detail               string        `json:"detail"`
}

// ---- the pure fold ------------------------------------------------------------

// Fold turns injected Facts into the full Status. It is pure: no git, no gh, no
// clock, no I/O. Every classification (dirty relevance, stable evidence, candidate
// state, next-action, loop-status) is a deterministic function of Facts, so the
// same Facts always yields the same Status.
func Fold(f Facts) Status {
	dirty := foldDirty(f.Dirty)
	stable := foldStable(f)
	action := foldNextAction(f.Decision, stable, dirty, f.CIDiagnosis)
	ok, verdict, detail := loopStatus(action)
	return Status{
		Schema:               Schema,
		Root:                 f.Root,
		HeadSHA:              f.HeadSHA,
		Branch:               f.Branch,
		Dirty:                dirty,
		LastTag:              f.LastTag,
		CommitsSinceTag:      f.CommitsSinceTag,
		FilesTouchedSinceTag: f.FilesTouchedSinceTag,
		Decision:             f.Decision.Decision,
		DecisionReason:       f.Decision.Reason,
		Cadence:              foldCadence(f.Cadence),
		Stable:               stable,
		NextAction:           action,
		OK:                   ok,
		Verdict:              verdict,
		Detail:               detail,
	}
}

// FoldBranchRegime turns branch-role/git/CI/lock facts into the dev/main drift view.
func FoldBranchRegime(f BranchRegimeFacts) BranchRegime {
	devBranch := branchDefault(f.DevelopmentBranch, "main")
	releaseBranch := branchDefault(f.ReleaseBranch, "main")
	out := BranchRegime{
		DevelopmentBranch: devBranch,
		DevelopmentHead:   strings.TrimSpace(f.DevelopmentHead),
		ReleaseBranch:     releaseBranch,
		ReleaseHead:       strings.TrimSpace(f.ReleaseHead),
		ReleaseSource:     strings.TrimSpace(f.ReleaseSource),
		PublicFrontDoor:   strings.TrimSpace(f.PublicFrontDoor),
		LatestTag:         strings.TrimSpace(f.LatestTag),
		DevelopmentAhead:  f.DevelopmentAhead,
		ReleaseAhead:      f.ReleaseAhead,
		Drift:             branchRegimeDrift(f),
		DevelopmentCI:     strings.TrimSpace(f.DevelopmentCI),
		ReleaseLockHeld:   f.ReleaseLockHeld,
		RoleError:         strings.TrimSpace(f.RoleError),
	}
	out.PromotionBlockers = branchRegimeBlockers(out)
	out.PromotionBlocked = len(out.PromotionBlockers) > 0
	if out.Drift == "development_ahead" && !out.PromotionBlocked {
		out.PromotionCandidate = out.DevelopmentHead
	}
	out.NextAction = branchRegimeNextAction(out)
	return out
}

func branchDefault(value, fallback string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return fallback
	}
	return value
}

func branchRegimeDrift(f BranchRegimeFacts) string {
	switch {
	case strings.TrimSpace(f.DevelopmentHead) == "" || strings.TrimSpace(f.ReleaseHead) == "":
		return "unknown"
	case f.DevelopmentAhead > 0 && f.ReleaseAhead > 0:
		return "diverged"
	case f.ReleaseAhead > 0:
		return "release_ahead"
	case f.DevelopmentAhead > 0:
		return "development_ahead"
	default:
		return "no_drift"
	}
}

func branchRegimeBlockers(r BranchRegime) []string {
	var blockers []string
	if r.RoleError != "" {
		blockers = append(blockers, "BRANCH_ROLE_CONFIG")
	}
	if r.DevelopmentHead == "" {
		blockers = append(blockers, "DEVELOPMENT_HEAD_UNKNOWN")
	}
	if r.ReleaseHead == "" {
		blockers = append(blockers, "RELEASE_HEAD_UNKNOWN")
	}
	if r.ReleaseAhead > 0 {
		blockers = append(blockers, "RELEASE_AHEAD")
	}
	if branchRegimeCIBlocks(r.DevelopmentCI) {
		blockers = append(blockers, "DEVELOPMENT_CI_RED")
	}
	if r.ReleaseLockHeld {
		blockers = append(blockers, "RELEASE_LOCK_HELD")
	}
	return nonNil(blockers)
}

func branchRegimeCIBlocks(status string) bool {
	switch strings.ToLower(strings.TrimSpace(status)) {
	case "failure", "failed", "red", "cancelled", "timed_out", "action_required":
		return true
	default:
		return false
	}
}

func branchRegimeNextAction(r BranchRegime) string {
	if r.PromotionBlocked {
		return "hold promotion; clear blocker(s): " + strings.Join(r.PromotionBlockers, ", ")
	}
	switch r.Drift {
	case "development_ahead":
		return fmt.Sprintf("promotion candidate: %s %s is %d commit(s) ahead of %s", r.DevelopmentBranch, shortSHA(r.DevelopmentHead), r.DevelopmentAhead, r.ReleaseBranch)
	case "no_drift":
		return fmt.Sprintf("hold; %s and %s point at the same release source", r.DevelopmentBranch, r.ReleaseBranch)
	case "unknown":
		return "refresh branch heads before deciding promotion status"
	default:
		return "inspect branch divergence before promotion"
	}
}

// foldDirty ports release_status.py dirty_summary() + is_release_relevant_dirty_path().
func foldDirty(entries []DirtyEntry) DirtySummary {
	var modified, untracked, relevant, unrelated []string
	for _, e := range entries {
		if e.Untracked {
			untracked = append(untracked, e.Path)
		} else {
			modified = append(modified, e.Path)
		}
	}
	for _, e := range entries {
		if isReleaseRelevantDirtyPath(e.Path) {
			relevant = append(relevant, e.Path)
		} else {
			unrelated = append(unrelated, e.Path)
		}
	}
	return DirtySummary{
		Clean:                len(entries) == 0,
		ModifiedCount:        len(modified),
		UntrackedCount:       len(untracked),
		Modified:             nonNil(modified),
		Untracked:            nonNil(untracked),
		ReleaseRelevantCount: len(relevant),
		ReleaseRelevant:      nonNil(relevant),
		UnrelatedCount:       len(unrelated),
		Unrelated:            nonNil(unrelated),
	}
}

// isReleaseRelevantDirtyPath ports release_status.py.is_release_relevant_dirty_path.
func isReleaseRelevantDirtyPath(path string) bool {
	normalized := strings.ReplaceAll(path, "\\", "/")
	if releaseRelevantExact[normalized] {
		return true
	}
	for _, prefix := range releaseRelevantPrefixes {
		if strings.HasPrefix(normalized, prefix) {
			return true
		}
	}
	return false
}

// dirtyRequiresCleanBeforeStatus ports release_status.py.dirty_requires_clean_before_status.
func dirtyRequiresCleanBeforeStatus(d DirtySummary) bool {
	if d.Clean {
		return false
	}
	return d.ReleaseRelevantCount > 0
}

// foldCadence ports release_status.py cadence_status() over the injected yaml text.
func foldCadence(c CadenceText) Cadence {
	path := c.Path
	if path == "" {
		path = ".github/workflows/release-cadence.yml"
	}
	if !c.Present {
		return Cadence{Present: false, Path: path}
	}
	text := c.Text
	manualGate := strings.Contains(text, "inputs.dry_run == false") ||
		strings.Contains(text, `inputs.dry_run }}" = "false"`)
	return Cadence{
		Present:              true,
		Path:                 path,
		Schedule:             strings.Contains(text, "schedule:"),
		ManualDispatch:       strings.Contains(text, "workflow_dispatch:"),
		DryRunFirst:          strings.Contains(text, "dry_run:") && manualGate,
		SingleWriter:         strings.Contains(text, "group: release-cadence") && strings.Contains(text, "cancel-in-progress: false"),
		TagAfterGreen:        strings.Contains(text, "tools/release_tag.py") && strings.Contains(text, "--require-ci") && strings.Contains(text, "--wait-ci"),
		CheckedGitHubRelease: strings.Contains(text, "tools/release_publish.py") && strings.Contains(text, "--execute"),
	}
}

// foldStable ports release_status.py stable_summary() over injected StableTags.
func foldStable(f Facts) StableSummary {
	evidence := foldStableEvidence(f.Stable)
	candidate := foldStableCandidate(f)
	if len(f.Stable) == 0 {
		return StableSummary{
			LatestStable:   nil,
			StableLag:      nil,
			Evidence:       evidence,
			Candidate:      candidate,
			Recommendation: stableRecommendation(nil, candidate, true),
		}
	}
	latest := f.Stable[0]
	latestCopy := latest
	stableTuple, ok := parseSemver(latest.Version)
	if !ok {
		return StableSummary{
			LatestStable:   &latestCopy,
			StableLag:      nil,
			Evidence:       evidence,
			Candidate:      candidate,
			Recommendation: "latest stable tag has no readable VERSION; inspect before promoting another stable",
		}
	}
	var newer []string
	for _, tag := range f.RollingTags {
		if t, ok := parseSemver(tag); ok && semverGreater(t, stableTuple) {
			newer = append(newer, tag)
		}
	}
	lag := len(newer)
	return StableSummary{
		LatestStable:     &latestCopy,
		StableLag:        &lag,
		NewerRollingTags: newer,
		Evidence:         evidence,
		Candidate:        candidate,
		Recommendation:   stableRecommendation(&lag, candidate, false),
	}
}

// foldStableEvidence ports release_status.py stable_evidence_status().
func foldStableEvidence(stable []StableTag) StableEvidence {
	var rows []StableEvidenceRow
	var failures []StableEvidenceFailure
	for _, tag := range stable {
		codename := tag.Tag
		if i := strings.Index(tag.Tag, "/"); i >= 0 {
			codename = tag.Tag[i+1:]
		}
		evidenceRel := "docs/stable-releases/" + codename + ".md"
		var issues []string
		if !tag.EvidenceFound {
			issues = append(issues, "missing committed evidence file "+evidenceRel)
		} else if len(tag.Evidence) == 0 {
			issues = append(issues, evidenceRel+" at HEAD has missing or unparseable frontmatter")
		}

		fields := tag.Evidence
		hasFields := tag.EvidenceFound && len(fields) > 0
		expectedSHA := tag.SHA
		if hasFields {
			candidateSHA := fields["candidate_sha"]
			if candidateSHA == "" {
				issues = append(issues, "frontmatter.candidate_sha missing")
			} else if !shaMatches(candidateSHA, expectedSHA) {
				issues = append(issues, fmt.Sprintf("frontmatter.candidate_sha=%q does not match %s commit %s", candidateSHA, tag.Tag, shortSHA(expectedSHA)))
			}

			underlying := fields["underlying_version"]
			if underlying == "" {
				issues = append(issues, "frontmatter.underlying_version missing")
			} else if _, ok := parseSemver(underlying); !ok {
				issues = append(issues, fmt.Sprintf("frontmatter.underlying_version=%q is not vX.Y.Z", underlying))
			} else if !tag.UnderlyingExists {
				issues = append(issues, "underlying rolling tag "+underlying+" does not exist")
			} else if !shaMatches(tag.UnderlyingTagSHA, expectedSHA) {
				issues = append(issues, fmt.Sprintf("underlying rolling tag %s points at %s, not %s", underlying, shortSHA(tag.UnderlyingTagSHA), shortSHA(expectedSHA)))
			}

			if cn := fields["codename"]; cn != "" && cn != codename {
				issues = append(issues, fmt.Sprintf("frontmatter.codename=%q does not match %q", cn, codename))
			}
		}

		rows = append(rows, StableEvidenceRow{
			Tag:          tag.Tag,
			EvidencePath: evidenceRel,
			OK:           len(issues) == 0,
			Issues:       nonNil(issues),
		})
		for _, issue := range issues {
			failures = append(failures, StableEvidenceFailure{Tag: tag.Tag, EvidencePath: evidenceRel, Detail: issue})
		}
	}
	return StableEvidence{
		OK:       len(failures) == 0,
		Checked:  len(stable),
		Failures: nonNilFailures(failures),
		Rows:     nonNilRows(rows),
	}
}

// foldStableCandidate ports release_status.py stable_candidate_status().
func foldStableCandidate(f Facts) StableCandidate {
	window := f.StableWindowDays
	if len(f.RollingTags) == 0 {
		return StableCandidate{
			Ready:      false,
			State:      "no_candidate",
			Reason:     "no reachable rolling vX.Y.Z tag exists",
			WindowDays: window,
		}
	}
	candidateTag := f.RollingTags[len(f.RollingTags)-1]
	candidateSHA := f.CandidateSHA
	row := StableCandidate{
		CandidateTag: candidateTag,
		CandidateSHA: candidateSHA,
		WindowDays:   window,
		Ready:        false,
	}
	if candidateSHA == "" {
		row.State = "unresolved"
		row.Reason = "could not resolve " + candidateTag
		return row
	}
	var promoted []string
	for _, item := range f.Stable {
		if shaMatches(item.SHA, candidateSHA) {
			promoted = append(promoted, item.Tag)
		}
	}
	if len(promoted) > 0 {
		row.State = "already_promoted"
		row.StableTag = promoted[0]
		row.Reason = candidateTag + " is already promoted as " + promoted[0]
		return row
	}
	if !f.CandidateAgeKnown {
		row.State = "unknown_age"
		row.Reason = "could not read " + candidateTag + " commit time"
		return row
	}
	age := round2(f.CandidateAgeDays)
	row.AgeDays = age
	row.SuggestedCodename = f.SuggestedCodename
	if age < window {
		row.State = "soaking"
		row.RemainingDays = round2(window - age)
		row.Reason = fmt.Sprintf("%s has soaked %sd of %sd", candidateTag, num(age), num(window))
		return row
	}
	row.Ready = true
	row.State = "ready"
	row.Reason = fmt.Sprintf("%s has soaked %sd; stable promotion preflight is due", candidateTag, num(age))
	return row
}

// stableRecommendation ports release_status.py stable_recommendation().
func stableRecommendation(lag *int, candidate StableCandidate, noStable bool) string {
	switch candidate.State {
	case "ready":
		return fmt.Sprintf("promote %s with /stable-release --codename %s", candidate.CandidateTag, candidate.SuggestedCodename)
	case "soaking":
		prefix := "no stable tag exists"
		if !noStable && lag != nil {
			prefix = fmt.Sprintf("stable lags by %d rolling release(s)", *lag)
		}
		return prefix + "; " + candidate.Reason
	case "already_promoted":
		return "stable channel is current"
	}
	if noStable {
		return "no stable tag exists; promote a soaked rolling tag when one is known good"
	}
	if lag != nil && *lag == 0 {
		return "stable channel is current"
	}
	lagN := 0
	if lag != nil {
		lagN = *lag
	}
	return fmt.Sprintf("stable lags by %d rolling release(s); consider /stable-release after soak", lagN)
}

// foldNextAction ports release_status.py next_action(): the single operator move.
func foldNextAction(decision Decision, stable StableSummary, dirty DirtySummary, ci CIDiagnosis) NextAction {
	blockers := decision.Blockers
	if decision.Decision == "release" {
		if !dirty.Clean {
			return NextAction{
				Kind: "cut_release_hot_tree",
				Detail: fmt.Sprintf(
					"cut %s with `fak release ship --execute`; it uses a detached origin/main checkout and leaves this checkout's %d modified and %d untracked path(s) untouched",
					decision.NextVersion, dirty.ModifiedCount, dirty.UntrackedCount),
			}
		}
		return NextAction{
			Kind:   "cut_release",
			Detail: fmt.Sprintf("cut %s with release_cut, push, then tag after green CI", decision.NextVersion),
		}
	}
	if dirtyRequiresCleanBeforeStatus(dirty) {
		relevant := dirty.ReleaseRelevantCount
		if relevant == 0 {
			relevant = dirty.ModifiedCount + dirty.UntrackedCount
		}
		return NextAction{
			Kind:   "clean_worktree",
			Detail: fmt.Sprintf("commit, shelve, or remove %d release-relevant dirty path(s) before treating release status as trunk evidence", relevant),
		}
	}
	if contains(blockers, "CI_BASE_RED") {
		if ci.Action == "fix_ci_billing" {
			return NextAction{Kind: "fix_ci_billing", Detail: ci.Detail}
		}
		return NextAction{Kind: "fix_ci", Detail: "fix current main ci.yml failure before cutting a release"}
	}
	if contains(blockers, "CI_RETRY_TO_GREEN") {
		return NextAction{
			Kind:   "pause_auto_release",
			Detail: "latest green ci.yml run was a retry; set FAK_AUTO_RELEASE=0 or confirm a fresh green run before cutting a release",
		}
	}
	if contains(blockers, "CI_BASE_NONE") || contains(blockers, "CI_STATE_UNKNOWN") {
		return NextAction{Kind: "confirm_ci", Detail: "restore or confirm a green main ci.yml signal"}
	}
	if contains(blockers, "WORKFLOW_UNPARSEABLE") {
		return NextAction{Kind: "fix_workflow", Detail: "repair GitHub workflow YAML before release"}
	}
	if contains(blockers, "VERSION_DRIFT") || contains(blockers, "VERSION_BEHIND_REACHABLE_TAG") {
		return NextAction{Kind: "fix_version_topology", Detail: "reconcile VERSION and semver tag topology"}
	}
	if len(blockers) > 0 {
		detail := decision.Reason
		if detail == "" {
			detail = blockers[0]
		}
		return NextAction{Kind: "hold", Detail: detail}
	}
	if !stable.Evidence.OK {
		detail := "stable tag evidence is missing or inconsistent"
		if len(stable.Evidence.Failures) > 0 {
			detail = fmt.Sprintf("repair %s: %s", stable.Evidence.Failures[0].Tag, stable.Evidence.Failures[0].Detail)
		}
		return NextAction{Kind: "repair_stable_evidence", Detail: detail}
	}
	if stable.Candidate.Ready {
		return NextAction{
			Kind:   "promote_stable",
			Detail: fmt.Sprintf("promote %s with /stable-release --codename %s", stable.Candidate.CandidateTag, stable.Candidate.SuggestedCodename),
		}
	}
	if stable.StableLag != nil && *stable.StableLag != 0 {
		return NextAction{Kind: "consider_stable", Detail: stable.Recommendation}
	}
	detail := decision.Reason
	if detail == "" {
		detail = "nothing release-worthy pending"
	}
	return NextAction{Kind: "wait", Detail: detail}
}

// loopStatus ports release_status.py loop_status_fields(): the (ok, verdict, detail)
// the loop reads — OK unless the next-action kind is in the action set.
func loopStatus(action NextAction) (bool, string, string) {
	kind := action.Kind
	if kind == "" {
		kind = "unknown"
	}
	detail := action.Detail
	if detail == "" {
		detail = kind
	}
	ok := !actionNextActions[kind]
	verdict := "OK"
	if !ok {
		verdict = "ACTION"
	}
	return ok, verdict, detail
}

// ---- semver + sha helpers (ported from release_status.py) ----------------------

func parseSemver(s string) ([3]int, bool) {
	s = strings.TrimSpace(s)
	if s == "" {
		return [3]int{}, false
	}
	if strings.HasPrefix(s, "v") {
		s = s[1:]
	}
	parts := strings.Split(s, ".")
	if len(parts) != 3 {
		return [3]int{}, false
	}
	var out [3]int
	for i, p := range parts {
		n, ok := atoiStrict(p)
		if !ok {
			return [3]int{}, false
		}
		out[i] = n
	}
	return out, true
}

func atoiStrict(s string) (int, bool) {
	if s == "" {
		return 0, false
	}
	n := 0
	for _, c := range s {
		if c < '0' || c > '9' {
			return 0, false
		}
		n = n*10 + int(c-'0')
	}
	return n, true
}

func semverGreater(a, b [3]int) bool {
	for i := 0; i < 3; i++ {
		if a[i] != b[i] {
			return a[i] > b[i]
		}
	}
	return false
}

func shaMatches(a, b string) bool {
	left := strings.ToLower(strings.TrimSpace(a))
	right := strings.ToLower(strings.TrimSpace(b))
	if left == "" || right == "" {
		return false
	}
	return left == right || strings.HasPrefix(left, right) || strings.HasPrefix(right, left)
}

func shortSHA(s string) string {
	s = strings.TrimSpace(s)
	if len(s) > 12 {
		return s[:12]
	}
	return s
}

func contains(values []string, want string) bool {
	for _, v := range values {
		if v == want {
			return true
		}
	}
	return false
}

func nonNil(s []string) []string {
	if s == nil {
		return []string{}
	}
	return s
}

func nonNilFailures(s []StableEvidenceFailure) []StableEvidenceFailure {
	if s == nil {
		return []StableEvidenceFailure{}
	}
	return s
}

func nonNilRows(s []StableEvidenceRow) []StableEvidenceRow {
	if s == nil {
		return []StableEvidenceRow{}
	}
	return s
}

// round2 rounds to two decimals to match the python round(..., 2) on age/remaining.
func round2(v float64) float64 {
	return float64(int64(v*100+0.5*sign(v))) / 100
}

func sign(v float64) float64 {
	if v < 0 {
		return -1
	}
	return 1
}

// num renders a number compactly: integral without a decimal, else with the
// minimal decimals, so "3" and "0.5" both read cleanly (matching the python which
// prints round()'d floats that often collapse to ints).
func num(v float64) string {
	if v == float64(int64(v)) {
		return fmt.Sprintf("%d", int64(v))
	}
	return strings.TrimRight(strings.TrimRight(fmt.Sprintf("%.2f", v), "0"), ".")
}

// ---- human render --------------------------------------------------------------

// Render is the one-screen human view, matching release_status.py render_human():
// the rolling decision verb + reason head line, last tag, commits-since-tag, the
// next-action, and the stable line.
func Render(s Status) string {
	decision := s.Decision
	if decision == "" {
		decision = "unknown"
	}
	lines := []string{
		fmt.Sprintf("release-status: %s - %s", strings.ToUpper(decision), s.DecisionReason),
		"  last tag: " + dashIfEmpty(s.LastTag),
		fmt.Sprintf("  commits since tag: %d", s.CommitsSinceTag),
		fmt.Sprintf("  next action: %s - %s", s.NextAction.Kind, s.NextAction.Detail),
	}
	if s.Stable.LatestStable != nil {
		latest := s.Stable.LatestStable
		lag := "(none)"
		if s.Stable.StableLag != nil {
			lag = fmt.Sprintf("%d", *s.Stable.StableLag)
		}
		lines = append(lines, fmt.Sprintf("  stable: %s (%s); lag=%s", latest.Tag, latest.Version, lag))
	} else {
		lines = append(lines, "  stable: none")
	}
	return strings.Join(lines, "\n")
}

func dashIfEmpty(s string) string {
	if strings.TrimSpace(s) == "" {
		return "(none)"
	}
	return s
}

// SortedRollingTags returns the rolling tags ascending by semver — a small helper a
// caller can use to normalize a tag list gathered out of order before folding.
func SortedRollingTags(tags []string) []string {
	out := append([]string(nil), tags...)
	sort.SliceStable(out, func(i, j int) bool {
		a, aok := parseSemver(out[i])
		b, bok := parseSemver(out[j])
		if aok && bok {
			return semverGreater(b, a) // ascending
		}
		return out[i] < out[j]
	})
	return out
}
