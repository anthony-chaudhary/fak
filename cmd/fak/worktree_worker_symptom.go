package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/witness"
	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

func isFixSubject(s string) bool {
	lower := strings.ToLower(strings.TrimSpace(s))
	return strings.HasPrefix(lower, "fix(") || strings.HasPrefix(lower, "fix:")
}

func workerLandSymptomUnwitnessed(detail string) workerworktree.Result {
	return workerworktree.Result{
		OK:     false,
		Code:   "SYMPTOM_UNWITNESSED",
		Reason: "SYMPTOM_UNWITNESSED: fix commit must include a test that fails on parent and passes on fix (red-then-green); bypass only with --unsafe-skip-symptom-witness",
		Detail: detail,
	}
}

// workerLandSymptomPreflight refuses a fix(*) land whose explicit paths carry no
// changed test before any candidate is materialized or built: the witness would
// refute it as SYMPTOM_NO_TEST anyway, only after the full prepare. An unscoped
// land (no --paths) is left to the witness, which reads the candidate commit.
func workerLandSymptomPreflight(paths []string) (workerworktree.Result, bool) {
	if len(paths) == 0 || len(witness.SymptomTestFiles(paths)) > 0 {
		return workerworktree.Result{}, false
	}
	res := workerLandSymptomUnwitnessed(symptomRefusalDetail("refuted", "no changed test file in the land paths (preflight, nothing built); "+symptomNoTestCure))
	res.SymptomVerdict = symptomSubtype("no changed test file")
	return res, true
}

const symptomNoTestCure = "cure: add a *_test.go (or test_*.py) to the land paths that fails on the parent and passes on the fix, driving the bug through API the parent already has; if there is no pre-existing symptom, retype the subject as feat/refactor/chore"

// splitTagList parses the comma-separated --symptom-tags value into a tag slice: trim each
// entry, drop empties. An unset flag yields nil, preserving today's untagged default.
func splitTagList(s string) []string {
	if strings.TrimSpace(s) == "" {
		return nil
	}
	var tags []string
	for _, part := range strings.Split(s, ",") {
		if part = strings.TrimSpace(part); part != "" {
			tags = append(tags, part)
		}
	}
	return tags
}

// defaultWorkerLandSymptomTimeout bounds the mandatory red-then-green symptom witness: two
// scratch worktrees plus a candidate and a parent `go test` compile. --symptom-timeout
// overrides it for large packages or a loaded host.
const defaultWorkerLandSymptomTimeout = 10 * time.Minute

func verifyWorkerLandSymptom(wtPath, ref string, extraTags []string, timeout time.Duration) workerworktree.Result {
	return verifyWorkerLandSymptomSelected(wtPath, ref, extraTags, nil, timeout)
}

func verifyWorkerLandSymptomSelected(wtPath, ref string, extraTags, testSelectors []string, timeout time.Duration) workerworktree.Result {
	if timeout <= 0 {
		timeout = defaultWorkerLandSymptomTimeout
	}
	resolver := witness.NewWithRunner(nil, wtPath).WithSymptomTags(extraTags).WithSymptomTests(testSelectors)
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	var snapshots []workerworktree.WitnessSnapshotEvent
	ctx = workerworktree.WithWitnessSnapshotObserver(ctx, func(event workerworktree.WitnessSnapshotEvent) {
		if len(snapshots) < 8 {
			snapshots = append(snapshots, event)
		}
	})
	ctx = witness.WithExecutionMaterializer(ctx, workerworktree.AcquireWitnessSnapshot)
	outcome, detail := resolver.ResolveSymptomWithDetail(ctx, ref, true)
	res := workerLandSymptomVerdict(outcome, detail, timeout)
	if len(snapshots) > 0 {
		if evidence, err := json.Marshal(snapshots); err == nil {
			res.Detail += " | witness_snapshots=" + string(evidence)
		}
	}
	if excerpt := resolver.SymptomFailureExcerpt(); !res.OK && excerpt != "" {
		res.Detail += " | " + excerpt
	}
	return res
}

// Closed symptom_verdict tokens. Only a refutation refuses the land; an abstention
// is not evidence against the fix (timeouts on a loaded host, a regression that
// names fix-introduced API), so it lands with a SYMPTOM_ABSTAIN_* advisory.
const (
	symptomVerdictConfirmed          = "SYMPTOM_CONFIRMED"
	symptomVerdictConfirmedTolerant  = "SYMPTOM_CONFIRMED_TOLERANT"
	symptomVerdictAbstainTimeout     = "SYMPTOM_ABSTAIN_TIMEOUT"
	symptomVerdictAbstainParentBuild = "SYMPTOM_ABSTAIN_PARENT_BUILD"
	symptomVerdictAbstainSelection   = "SYMPTOM_ABSTAIN_SELECTION"
	symptomVerdictAbstainOther       = "SYMPTOM_ABSTAIN_OTHER"
)

func workerLandSymptomVerdict(outcome abi.WitnessOutcome, detail string, timeout time.Duration) workerworktree.Result {
	switch outcome {
	case abi.WitnessConfirmed:
		res := workerworktree.Result{OK: true, SymptomVerdict: symptomVerdictConfirmed}
		// The verdict stays SYMPTOM_CONFIRMED; the tolerance is an advisory so the
		// closed verdict set and its consumers do not change.
		if strings.Contains(detail, "tolerated at candidate") {
			res.SymptomAdvisory = symptomVerdictConfirmedTolerant + ": symptom witness confirmed with tolerated candidate reds (not cached): " + detail
		}
		return res
	case abi.WitnessRefuted:
		res := workerLandSymptomUnwitnessed(symptomRefusalDetail("refuted", detail))
		res.SymptomVerdict = symptomSubtype(detail)
		return res
	default:
		token := symptomAbstainToken(detail)
		advisory := token + ": symptom witness abstained, landing without red-then-green proof: " + detail
		if token == symptomVerdictAbstainTimeout {
			advisory += fmt.Sprintf(" [symptom budget %s; raise with --symptom-timeout]", timeout)
		}
		return workerworktree.Result{OK: true, SymptomVerdict: token, SymptomAdvisory: advisory}
	}
}

// symptomAbstainToken classifies an abstention. A timeout is classified first: a run
// the budget killed can leave a detail that also names a parent failure.
func symptomAbstainToken(detail string) string {
	switch symptomSubtype(detail) {
	case "SYMPTOM_TIMEOUT":
		return symptomVerdictAbstainTimeout
	case "SYMPTOM_PARENT_BUILD":
		return symptomVerdictAbstainParentBuild
	case "SYMPTOM_SELECTION_AMBIGUOUS":
		return symptomVerdictAbstainSelection
	}
	if strings.Contains(detail, "selector") || strings.Contains(detail, "no executable package") {
		return symptomVerdictAbstainSelection
	}
	return symptomVerdictAbstainOther
}

// symptomRefusalDetail keeps the established top-level SYMPTOM_UNWITNESSED code
// while exposing a stable subtype for automation and operator remediation.
func symptomRefusalDetail(outcome, detail string) string {
	return symptomSubtype(detail) + ": symptom witness was " + outcome + ": " + detail
}

// symptomSubtype maps a witness detail onto its closed subtype. A timeout is
// classified first: a run the budget killed can leave a detail that also names a
// zero match or a parent failure, and neither is evidence once the budget expired.
func symptomSubtype(detail string) string {
	switch {
	case strings.Contains(detail, "candidate") && strings.Contains(detail, "did not build"):
		return "SYMPTOM_CANDIDATE_BUILD"
	case strings.Contains(detail, "timed out"):
		return "SYMPTOM_TIMEOUT"
	case strings.Contains(detail, "matched no"):
		return "SYMPTOM_NO_MATCH"
	case strings.Contains(detail, "parent") && (strings.Contains(detail, "did not build") || strings.Contains(detail, "did not parse")):
		return "SYMPTOM_PARENT_BUILD"
	// Must precede "passed at parent": a regression detail names that phrase too.
	case strings.Contains(detail, "symptom test regressed"):
		return "SYMPTOM_CANDIDATE_REGRESSION"
	case strings.Contains(detail, "passed at parent"):
		return "SYMPTOM_PARENT_PASSED"
	case strings.Contains(detail, "candidate") && strings.Contains(detail, "failed"):
		return "SYMPTOM_CANDIDATE_FAILED"
	case strings.Contains(detail, "ambiguous"):
		return "SYMPTOM_SELECTION_AMBIGUOUS"
	case strings.Contains(detail, "no changed test file"):
		return "SYMPTOM_NO_TEST"
	}
	return "SYMPTOM_OTHER"
}

// withSymptomVerdict stamps the witness verdict onto the land result and prints a
// successful land's abstention as one stable `advisory: SYMPTOM_ABSTAIN_*` line.
func withSymptomVerdict(res, symptom workerworktree.Result, stderr io.Writer) workerworktree.Result {
	if symptom.SymptomVerdict == "" {
		return res
	}
	if res.SymptomVerdict == "" {
		res.SymptomVerdict = symptom.SymptomVerdict
	}
	if res.OK && symptom.SymptomAdvisory != "" {
		res.SymptomAdvisory = symptom.SymptomAdvisory
		fmt.Fprintln(stderr, "advisory: "+symptom.SymptomAdvisory)
	}
	return res
}
