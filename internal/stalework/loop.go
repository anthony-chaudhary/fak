package stalework

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/issuecohort"
	"github.com/anthony-chaudhary/fak/internal/issuepolicy"
)

const (
	LoopSchema  = "fak.stale-work.loop-plan.v1"
	StateSchema = "fak.stale-work.loop-state.v1"

	DispatchReady  = "READY"
	DispatchRefuse = "REFUSE"

	ReconcileShipped   = "SHIPPED"
	ReconcileStillOpen = "STILL_OPEN"
	ReconcileAbstain   = "ABSTAIN"

	ReasonIssueRequired          = "STALE_WORK_ISSUE_REQUIRED"
	ReasonIssueContractInvalid   = "STALE_WORK_ISSUE_CONTRACT_INVALID"
	ReasonIssueDedupeAmbiguous   = "STALE_WORK_ISSUE_DEDUPE_AMBIGUOUS"
	ReasonWorkerIdentityReused   = "STALE_WORK_WORKER_ID_REUSED"
	ReasonAdjudicationCached     = "STALE_WORK_ADJUDICATION_CACHE_REUSED"
	ReasonAdjudicationInvalid    = "STALE_WORK_ADJUDICATION_EVIDENCE_CHANGED"
	ReasonIssueStillOpen         = "STALE_WORK_ISSUE_STILL_OPEN"
	ReasonAlreadyShipped         = "STALE_WORK_ALREADY_SHIPPED"
	ReasonIssueWitnessMissing    = "STALE_WORK_ISSUE_WITNESS_MISSING"
	ReasonCommitWitnessMissing   = "STALE_WORK_COMMIT_WITNESS_MISSING"
	ReasonTestWitnessMissing     = "STALE_WORK_TEST_WITNESS_MISSING"
	ReasonDecisionWitnessMissing = "STALE_WORK_DECISION_WITNESS_MISSING"
)

// IssueSnapshot is the deterministic issue read-back accepted by the loop. Callers
// should include open plus recently-closed issues so marker/path dedupe cannot create
// a replacement while a just-closed adjudication is still settling.
type IssueSnapshot struct {
	Number   int                      `json:"number"`
	Title    string                   `json:"title"`
	Body     string                   `json:"body"`
	State    string                   `json:"state"`
	ClosedAt string                   `json:"closedAt,omitempty"`
	URL      string                   `json:"url,omitempty"`
	Labels   []issuepolicy.IssueLabel `json:"labels,omitempty"`
}

type WitnessRecord struct {
	Issue     int    `json:"issue"`
	SHA       string `json:"sha,omitempty"`
	Verdict   string `json:"verdict,omitempty"`
	Witness   string `json:"witness,omitempty"`
	TestClaim string `json:"test_claim,omitempty"`
	Decision  string `json:"decision,omitempty"` // retained | updated | deleted
	Source    string `json:"source,omitempty"`   // independent read-back/ledger reference
}

type AdjudicationRecord struct {
	DedupeKey      string `json:"dedupe_key"`
	EvidenceDigest string `json:"evidence_digest"`
	Decision       string `json:"decision"`
	Reason         string `json:"reason,omitempty"`
	Issue          int    `json:"issue,omitempty"`
	Witness        string `json:"witness,omitempty"`
}

type LoopState struct {
	Schema        string               `json:"schema"`
	Adjudications []AdjudicationRecord `json:"adjudications"`
}

type IssuePlan struct {
	Action  string             `json:"action"`
	Number  int                `json:"number,omitempty"`
	State   string             `json:"state,omitempty"`
	URL     string             `json:"url,omitempty"`
	Title   string             `json:"title,omitempty"`
	Body    string             `json:"body,omitempty"`
	Command []string           `json:"command,omitempty"`
	Review  issuepolicy.Review `json:"contract"`
}

type DispatchPlan struct {
	Status   string   `json:"status"`
	Reason   string   `json:"reason,omitempty"`
	WorkerID string   `json:"worker_id,omitempty"`
	Wave     int      `json:"wave,omitempty"`
	Lane     string   `json:"lane,omitempty"`
	Paths    []string `json:"paths,omitempty"`
	Command  []string `json:"command,omitempty"`
	Launched bool     `json:"launched"`
	ExitCode int      `json:"exit_code,omitempty"`
}

type Reconciliation struct {
	Status    string `json:"status"`
	Reason    string `json:"reason"`
	Issue     int    `json:"issue,omitempty"`
	SHA       string `json:"sha,omitempty"`
	Decision  string `json:"decision,omitempty"`
	TestClaim string `json:"test_claim,omitempty"`
}

type IssueUnit struct {
	DedupeKey         string         `json:"dedupe_key"`
	EvidenceDigest    string         `json:"evidence_digest"`
	BatchKey          string         `json:"batch_key"`
	Path              string         `json:"path"`
	DiscoveryBatch    string         `json:"discovery_batch"`
	TruthSource       string         `json:"truth_source"`
	AcceptanceWitness string         `json:"acceptance_witness"`
	Cache             string         `json:"cache,omitempty"`
	CacheReason       string         `json:"cache_reason,omitempty"`
	Issue             IssuePlan      `json:"issue"`
	Dispatch          DispatchPlan   `json:"dispatch"`
	Reconciliation    Reconciliation `json:"reconciliation"`
}

type LoopBatch struct {
	Key               string   `json:"key"`
	TruthSource       string   `json:"truth_source"`
	AcceptanceWitness string   `json:"acceptance_witness"`
	Units             []string `json:"units"`
}

type LoopCounts struct {
	Candidates      int `json:"candidates"`
	IssueBound      int `json:"issue_bound"`
	CreatePlanned   int `json:"create_planned"`
	ContractInvalid int `json:"contract_invalid"`
	Cached          int `json:"cached"`
	DispatchReady   int `json:"dispatch_ready"`
	PlannedLaunches int `json:"planned_launches"`
	Launches        int `json:"launches"`
	CollisionPairs  int `json:"collision_pairs"`
	Waves           int `json:"waves"`
}

type LoopPlan struct {
	Schema        string               `json:"schema"`
	Mode          string               `json:"mode"`
	PacketHead    string               `json:"packet_head"`
	Units         []IssueUnit          `json:"units"`
	Batches       []LoopBatch          `json:"batches"`
	Waves         []issuecohort.Wave   `json:"waves,omitempty"`
	Adjudications []AdjudicationRecord `json:"adjudications"`
	NextState     LoopState            `json:"next_state"`
	Counts        LoopCounts           `json:"counts"`
}

type LoopOptions struct {
	Issues          []IssueSnapshot
	State           LoopState
	Witnesses       []WitnessRecord
	MaxWave         int
	LiveIssueCreate bool
	LiveLaunch      bool
}

// BuildLoop is pure: it renders effects, but performs none. The cmd shell may
// execute only commands explicitly armed by LiveIssueCreate/LiveLaunch.
func BuildLoop(packet Packet, opt LoopOptions) LoopPlan {
	mode := "dry-run"
	if opt.LiveIssueCreate || opt.LiveLaunch {
		mode = "live-requested"
	}
	plan := LoopPlan{Schema: LoopSchema, Mode: mode, PacketHead: packet.Head}
	candidates := append([]Candidate(nil), packet.Candidates...)
	sort.Slice(candidates, func(i, j int) bool { return candidates[i].DedupeKey < candidates[j].DedupeKey })
	plan.Counts.Candidates = len(candidates)

	state := stateIndex(opt.State)
	witnesses := witnessIndex(opt.Witnesses)
	next := state
	cohortCandidates := []issuepolicy.Candidate{}
	unitByKey := map[string]int{}

	for _, candidate := range candidates {
		digest := EvidenceDigest(candidate)
		unit := IssueUnit{
			DedupeKey: candidate.DedupeKey, EvidenceDigest: digest, Path: candidate.Path,
			DiscoveryBatch: candidate.Batch, TruthSource: candidate.Path,
			AcceptanceWitness: candidate.VerifyWith,
			Dispatch:          DispatchPlan{Status: DispatchRefuse, Reason: ReasonIssueRequired},
		}
		unit.BatchKey = batchKey(unit.TruthSource, unit.AcceptanceWitness)
		generated := contractCandidate(candidate, digest)
		title, body := renderIssue(generated, candidate, digest)
		labels := make([]issuepolicy.IssueLabel, len(generated.Labels))
		for i, l := range generated.Labels {
			labels[i] = issuepolicy.IssueLabel{Name: l}
		}
		genReview := issuepolicy.ReviewIssueDraft(issuepolicy.IssueDraft{Title: title, Body: body, Labels: labels}, issuepolicy.Options{StrictWitness: true})
		unit.Issue = IssuePlan{
			Action: "create", Title: title, Body: body, Review: genReview,
			Command: issueCreateCommand(title, body, !opt.LiveIssueCreate),
		}

		if prior, ok := state[candidate.DedupeKey]; ok {
			switch {
			case prior.EvidenceDigest != digest:
				unit.Cache = "invalidated"
				unit.CacheReason = ReasonAdjudicationInvalid
			case terminalDecision(prior.Decision) && strings.TrimSpace(prior.Witness) != "":
				unit.Cache = "reused"
				unit.Issue.Action = "cached"
				unit.Dispatch.Reason = ReasonAdjudicationCached
				unit.Reconciliation = Reconciliation{Status: ReconcileAbstain, Reason: ReasonAdjudicationCached, Issue: prior.Issue, Decision: prior.Decision}
				plan.Counts.Cached++
				plan.Adjudications = append(plan.Adjudications, prior)
				plan.Units = append(plan.Units, unit)
				continue
			}
		}

		matches := matchingIssues(candidate, digest, opt.Issues)
		switch len(matches) {
		case 0:
			plan.Counts.CreatePlanned++
			unit.Reconciliation = Reconciliation{Status: ReconcileAbstain, Reason: ReasonIssueRequired}
			rec := AdjudicationRecord{DedupeKey: candidate.DedupeKey, EvidenceDigest: digest, Decision: "issue_required", Reason: ReasonIssueRequired}
			plan.Adjudications = append(plan.Adjudications, rec)
			next[candidate.DedupeKey] = rec
		case 1:
			issue := matches[0]
			draft := issuepolicy.IssueDraft{Number: issue.Number, Title: issue.Title, Body: issue.Body, Labels: issue.Labels, URL: issue.URL}
			review := issuepolicy.ReviewIssueDraft(draft, issuepolicy.Options{StrictWitness: true})
			unit.Issue = IssuePlan{Action: "reuse", Number: issue.Number, State: issue.State, URL: issue.URL, Title: issue.Title, Body: issue.Body, Review: review}
			plan.Counts.IssueBound++
			if !review.OK || review.Dispatchability != issuepolicy.Dispatchable {
				unit.Issue.Action = "repair"
				unit.Dispatch.Reason = ReasonIssueContractInvalid
				unit.Reconciliation = reconcile(issue, witnesses[issue.Number])
				plan.Counts.ContractInvalid++
				rec := AdjudicationRecord{DedupeKey: candidate.DedupeKey, EvidenceDigest: digest, Decision: "contract_refused", Reason: ReasonIssueContractInvalid, Issue: issue.Number}
				plan.Adjudications = append(plan.Adjudications, rec)
				next[candidate.DedupeKey] = rec
				break
			}
			parsed := issuepolicy.CandidateFromIssueDraft(draft)
			parsed.Key = candidate.DedupeKey
			parsed.IssueNumber = issue.Number
			unit.Reconciliation = reconcile(issue, witnesses[issue.Number])
			if unit.Reconciliation.Status == ReconcileStillOpen {
				cohortCandidates = append(cohortCandidates, parsed)
				unitByKey[candidate.DedupeKey] = len(plan.Units)
			} else if unit.Reconciliation.Status == ReconcileShipped {
				unit.Dispatch.Reason = ReasonAlreadyShipped
			} else {
				unit.Dispatch.Reason = unit.Reconciliation.Reason
			}
			rec := adjudicationFromReconcile(candidate.DedupeKey, digest, unit.Reconciliation, witnesses[issue.Number])
			plan.Adjudications = append(plan.Adjudications, rec)
			next[candidate.DedupeKey] = rec
		default:
			unit.Issue.Action = "refuse"
			unit.Dispatch.Reason = ReasonIssueDedupeAmbiguous
			unit.Reconciliation = Reconciliation{Status: ReconcileAbstain, Reason: ReasonIssueDedupeAmbiguous}
			rec := AdjudicationRecord{DedupeKey: candidate.DedupeKey, EvidenceDigest: digest, Decision: "dedupe_refused", Reason: ReasonIssueDedupeAmbiguous}
			plan.Adjudications = append(plan.Adjudications, rec)
			next[candidate.DedupeKey] = rec
		}
		plan.Units = append(plan.Units, unit)
	}

	cohort := issuecohort.Build(cohortCandidates, issuecohort.Options{
		Options: issuepolicy.Options{StrictWitness: true}, MaxWave: opt.MaxWave,
	})
	plan.Waves = cohort.Waves
	plan.Counts.CollisionPairs = cohort.CollisionPairs
	plan.Counts.Waves = cohort.NumWaves
	seenWorkers := map[string]bool{}
	for _, wave := range cohort.Waves {
		for _, member := range wave.Members {
			idx, ok := unitByKey[member.Key]
			if !ok {
				continue
			}
			unit := &plan.Units[idx]
			leaseID := fmt.Sprintf("resolve-%s-%d", idToken(member.Lane), member.IssueNumber)
			enrollment := dispatchtick.PlanHostEnrollment(member.Lane, member.IssueNumber, leaseID, member.Paths)
			if seenWorkers[enrollment.AgentID] {
				unit.Dispatch = DispatchPlan{Status: DispatchRefuse, Reason: ReasonWorkerIdentityReused}
				continue
			}
			seenWorkers[enrollment.AgentID] = true
			cmd := []string{"fak", "dispatch", "tick", "--lane", member.Lane, "--target-issue", strconv.Itoa(member.IssueNumber),
				"--lease-id", enrollment.AgentID, "--lease-tree", strings.Join(member.Paths, ","), "--json"}
			if opt.LiveLaunch {
				cmd = append(cmd, "--live")
			}
			unit.Dispatch = DispatchPlan{
				Status: DispatchReady, WorkerID: enrollment.AgentID, Wave: wave.Index,
				Lane: member.Lane, Paths: append([]string(nil), member.Paths...), Command: cmd,
			}
			plan.Counts.DispatchReady++
			plan.Counts.PlannedLaunches++
		}
	}
	plan.NextState = LoopState{Schema: StateSchema, Adjudications: sortedState(next)}
	plan.Batches = buildBatches(plan.Units)
	sort.Slice(plan.Adjudications, func(i, j int) bool { return plan.Adjudications[i].DedupeKey < plan.Adjudications[j].DedupeKey })
	return plan
}

func EvidenceDigest(candidate Candidate) string {
	b, _ := json.Marshal(candidate)
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

func batchKey(truthSource, witness string) string {
	sum := sha256.Sum256([]byte(strings.TrimSpace(truthSource) + "\x00" + strings.TrimSpace(witness)))
	return "stale-work-batch:" + hex.EncodeToString(sum[:8])
}

func buildBatches(units []IssueUnit) []LoopBatch {
	index := map[string]*LoopBatch{}
	for _, unit := range units {
		batch := index[unit.BatchKey]
		if batch == nil {
			batch = &LoopBatch{Key: unit.BatchKey, TruthSource: unit.TruthSource, AcceptanceWitness: unit.AcceptanceWitness}
			index[unit.BatchKey] = batch
		}
		batch.Units = append(batch.Units, unit.DedupeKey)
	}
	out := make([]LoopBatch, 0, len(index))
	for _, batch := range index {
		sort.Strings(batch.Units)
		out = append(out, *batch)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Key < out[j].Key })
	return out
}

func staleWorkProblemFrame() issuepolicy.ProblemFrame {
	checks := map[string]issuepolicy.ProblemCheck{
		"p1": {ID: "p1", Status: issuepolicy.ProblemCheckAdvanced, Evidence: "the candidate preserves path, semantic commit, evidence digest, and score provenance", Valid: true},
		"p2": {ID: "p2", Status: issuepolicy.ProblemCheckAdvanced, Evidence: "one adjudication prevents repeated rediscovery while refusing unsupported mutation", Valid: true},
		"p3": {ID: "p3", Status: issuepolicy.ProblemCheckAdvanced, Evidence: "retain, update, or delete remains a bounded worker decision after fresh evidence", Valid: true},
		"p4": {ID: "p4", Status: issuepolicy.ProblemCheckAdvanced, Evidence: "the evidence-backed packet flows through cohort planning and witnessed closure", Valid: true},
	}
	return issuepolicy.ProblemFrame{Schema: issuepolicy.ProblemFrameSchema, Ready: true, Enforced: true, Centrality: issuepolicy.CentralityStewardship, CentralityTarget: "adjudicate stale tracked work without losing evidence or silently deleting value", Checks: checks}
}

func contractCandidate(c Candidate, digest string) issuepolicy.Candidate {
	why := make([]string, 0, len(c.Components))
	for _, component := range c.Components {
		why = append(why, fmt.Sprintf("%s=%d (%s: %s)", component.Name, component.Points, component.Provenance, component.Evidence))
	}
	if len(why) == 0 {
		why = append(why, "the stale-work packet supplied a supported semantic-drift candidate")
	}
	return issuepolicy.Candidate{
		Schema: issuepolicy.Schema, ProblemFrame: staleWorkProblemFrame(), Key: c.DedupeKey,
		Title: "stale-work: adjudicate " + c.Path, ParentRef: "#6618",
		CurrentState: fmt.Sprintf("Packet evidence digest `%s` names `%s`; last semantic commit `%s`.", digest, c.Path, c.LastSemanticCommit),
		WhyNow:       strings.Join(why, "; "),
		WorkingSpine: "Use this dedicated issue as the only authority for one fresh worker to adjudicate and, only after the decision, retain, update, or delete the candidate.",
		WorkUnit:     "leaf", ExpectedSteps: 5,
		Assumptions:    []string{"The packet is evidence for review, not authority to mutate the candidate."},
		ConfusionRisks: []string{"Stale-looking prose can still be intentionally historical or currently true."},
		Coordination:   []string{"Use a fresh worker identity dedicated to this issue; do not reuse the discovery session."},
		Trigger:        "supported stale-work candidate",
		BatchPolicy:    "one issue per stable dedupe key and artifact truth source; reuse an existing open or recently closed match",
		InScope:        fmt.Sprintf("Adjudicate `%s` against current source, record retain/update/delete, and implement only that issue-approved decision.", c.Path),
		OutOfScope:     "Do not edit any other stale-work candidate, expand the scanner, or close from worker narration.",
		DoneCondition:  "Record an explicit retained, updated, or deleted decision; if content changes, keep the change scoped to this candidate and its focused proof.",
		Witness:        c.VerifyWith + "; git read-back plus `dos commit-audit` and a green issue-specific test/read-back must independently confirm the result.",
		AcceptanceGate: "The issue remains open unless independent issue state, diff-witnessed commit, and green acceptance evidence agree.",
		Lane:           laneForPath(c.Path), Paths: []string{c.Path},
		Labels:             []string{"class:dev", "gen/now"},
		ClosureBinding:     "A resolving commit cites this dedicated issue; `dos commit-audit` returns OK/diff-witnessed, the acceptance witness is green, and an independent issue read-back is closed.",
		CompletionStandard: "development",
		WorkEstimate:       "Estimate: 1 points. Uncertainty: adjudication may choose retain with no content change.",
		ScopeContribution:  "Contribution: 1/100 points.",
	}
}

func renderIssue(c issuepolicy.Candidate, source Candidate, digest string) (string, string) {
	lines := []string{
		"<!-- fak-stale-work-key: " + c.Key + " -->",
		"",
		"## Parent context", c.ParentRef,
		"", "## Current state", c.CurrentState,
		"- Evidence digest: `" + digest + "`",
		"- Discovery batch: `" + source.Batch + "`",
		"", "## Why now", c.WhyNow,
		"", "## Working spine", c.WorkingSpine,
		"", "## Work unit", c.WorkUnit,
		"", "## Expected steps", strconv.Itoa(c.ExpectedSteps),
		"", "## Assumptions", "- " + c.Assumptions[0],
		"", "## Confusion risks", "- " + c.ConfusionRisks[0],
		"", "## Coordination", "- " + c.Coordination[0],
		"", "## Trigger", c.Trigger,
		"", "## Batch policy", c.BatchPolicy,
		"", "## Core through-line", c.InScope,
		"", "## Gold-plating boundary", c.OutOfScope,
		"", "## Problem frame", problemFrameLines(c.ProblemFrame),
		"", "## Definition of done",
		"- [ ] " + c.DoneCondition,
		"", "## Witness", c.Witness,
		"", "## Acceptance gate", c.AcceptanceGate,
		"", "## Lane", c.Lane,
		"", "## Likely files", "- `" + source.Path + "`",
		"", "## Closure binding", c.ClosureBinding,
		"", "## Work estimate", c.WorkEstimate,
		"", "## Overall completion contribution", c.ScopeContribution,
		"", "## Completion standard", c.CompletionStandard,
	}
	return c.Title, strings.Join(lines, "\n") + "\n"
}

func problemFrameLines(frame issuepolicy.ProblemFrame) string {
	centrality := frame.Centrality
	if frame.CentralityTarget != "" {
		centrality += " (" + frame.CentralityTarget + ")"
	}
	lines := []string{"- Centrality: " + centrality}
	for _, id := range []string{"p1", "p2", "p3", "p4"} {
		check := frame.Checks[id]
		lines = append(lines, "- "+strings.ToUpper(id)+": "+check.Status+" - "+check.Evidence)
	}
	return strings.Join(lines, "\n")
}

func issueCreateCommand(title, body string, dry bool) []string {
	cmd := []string{"fak", "issue", "create", "--title", title, "--body", body, "--labels", "class:dev,gen/now", "--json"}
	if dry {
		cmd = append(cmd, "--dry-run")
	}
	return cmd
}

func matchingIssues(candidate Candidate, digest string, issues []IssueSnapshot) []IssueSnapshot {
	var out []IssueSnapshot
	for _, issue := range issues {
		parsed := issuepolicy.CandidateFromIssueDraft(issuepolicy.IssueDraft{
			Number: issue.Number, Title: issue.Title, Body: issue.Body, Labels: issue.Labels, URL: issue.URL,
		})
		body := strings.ToLower(strings.ReplaceAll(issue.Body, "\\", "/"))
		if parsed.Key == candidate.DedupeKey ||
			containsPath(parsed.Paths, candidate.Path) ||
			bodyContainsExactPath(body, candidate.Path) ||
			strings.Contains(body, strings.ToLower(digest)) {
			out = append(out, issue)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Number < out[j].Number })
	return out
}

func bodyContainsExactPath(body, path string) bool {
	path = strings.ToLower(strings.Trim(strings.ReplaceAll(path, "\\", "/"), "/"))
	if path == "" {
		return false
	}
	return strings.Contains(body, "`"+path+"`") ||
		strings.Contains(body, "\n"+path+"\n") ||
		strings.HasSuffix(body, "\n"+path)
}

func reconcile(issue IssueSnapshot, witness WitnessRecord) Reconciliation {
	state := strings.ToUpper(strings.TrimSpace(issue.State))
	if state == "OPEN" {
		return Reconciliation{Status: ReconcileStillOpen, Reason: ReasonIssueStillOpen, Issue: issue.Number}
	}
	if state != "CLOSED" {
		return Reconciliation{Status: ReconcileAbstain, Reason: ReasonIssueWitnessMissing, Issue: issue.Number}
	}
	if !dispatchtick.CommitWitnessed(witness.Verdict, witness.Witness) || strings.TrimSpace(witness.SHA) == "" {
		return Reconciliation{Status: ReconcileAbstain, Reason: ReasonCommitWitnessMissing, Issue: issue.Number}
	}
	if witness.TestClaim != dispatchtick.ClaimTestGreen {
		return Reconciliation{Status: ReconcileAbstain, Reason: ReasonTestWitnessMissing, Issue: issue.Number, SHA: witness.SHA, TestClaim: witness.TestClaim}
	}
	decision := strings.ToLower(strings.TrimSpace(witness.Decision))
	if !terminalDecision(decision) || strings.TrimSpace(witness.Source) == "" {
		return Reconciliation{Status: ReconcileAbstain, Reason: ReasonDecisionWitnessMissing, Issue: issue.Number, SHA: witness.SHA, TestClaim: witness.TestClaim}
	}
	return Reconciliation{Status: ReconcileShipped, Reason: "independent_issue_git_test_witnesses", Issue: issue.Number, SHA: witness.SHA, Decision: decision, TestClaim: witness.TestClaim}
}

func adjudicationFromReconcile(key, digest string, r Reconciliation, w WitnessRecord) AdjudicationRecord {
	decision := "still_open"
	if r.Status == ReconcileShipped {
		decision = r.Decision
	} else if r.Status == ReconcileAbstain {
		decision = "abstain"
	}
	return AdjudicationRecord{DedupeKey: key, EvidenceDigest: digest, Decision: decision, Reason: r.Reason, Issue: r.Issue, Witness: w.Source}
}

func terminalDecision(s string) bool {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "valid", "retained", "updated", "deleted":
		return true
	default:
		return false
	}
}

func stateIndex(state LoopState) map[string]AdjudicationRecord {
	out := map[string]AdjudicationRecord{}
	for _, rec := range state.Adjudications {
		if strings.TrimSpace(rec.DedupeKey) != "" {
			out[rec.DedupeKey] = rec
		}
	}
	return out
}

func witnessIndex(in []WitnessRecord) map[int]WitnessRecord {
	out := map[int]WitnessRecord{}
	for _, row := range in {
		if row.Issue > 0 {
			out[row.Issue] = row
		}
	}
	return out
}

func sortedState(index map[string]AdjudicationRecord) []AdjudicationRecord {
	out := make([]AdjudicationRecord, 0, len(index))
	for _, rec := range index {
		out = append(out, rec)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].DedupeKey < out[j].DedupeKey })
	return out
}

func containsPath(paths []string, want string) bool {
	want = strings.ToLower(strings.Trim(strings.ReplaceAll(want, "\\", "/"), "/"))
	for _, path := range paths {
		if strings.ToLower(strings.Trim(strings.ReplaceAll(path, "\\", "/"), "/")) == want {
			return true
		}
	}
	return false
}

func laneForPath(path string) string {
	path = strings.Trim(strings.ReplaceAll(path, "\\", "/"), "/")
	switch {
	case path == "CLAIMS.md":
		return "claims"
	case strings.HasPrefix(path, "docs/"):
		return "docs"
	case strings.HasPrefix(path, ".claude/"):
		return "claude"
	case strings.HasPrefix(path, "cmd/"):
		return "cmd"
	case strings.HasPrefix(path, "internal/"):
		parts := strings.Split(path, "/")
		if len(parts) > 1 {
			return parts[1]
		}
	}
	if part, _, ok := strings.Cut(path, "/"); ok && part != "" {
		return part
	}
	return "docs"
}

func idToken(s string) string {
	var b strings.Builder
	for _, r := range strings.TrimSpace(s) {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '_':
			b.WriteRune(r)
		default:
			b.WriteByte('-')
		}
	}
	if b.Len() == 0 {
		return "unknown"
	}
	return b.String()
}
