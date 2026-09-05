package main

import (
	"bytes"
	"io"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestRecoverOffTrunkDryRunPrintsCommands(t *testing.T) {
	var out, errb bytes.Buffer
	if rc := runRecover(&out, &errb, []string{"OFF_TRUNK", "--dry-run", "--trunk", "main"}); rc != 0 {
		t.Fatalf("rc = %d, stderr=%s", rc, errb.String())
	}
	got := out.String()
	for _, want := range []string{"recover OFF_TRUNK", "git fetch origin main", "git merge --no-edit origin/main", "never force-push"} {
		if !strings.Contains(got, want) {
			t.Fatalf("dry-run output missing %q:\n%s", want, got)
		}
	}
}

func TestRecoverOffTrunkExecuteRunsSafeSteps(t *testing.T) {
	old := recoverRunStep
	t.Cleanup(func() { recoverRunStep = old })
	var ran [][]string
	recoverRunStep = func(dir string, argv []string, stdout, stderr io.Writer) int {
		ran = append(ran, append([]string(nil), argv...))
		return 0
	}

	var out, errb bytes.Buffer
	if rc := runRecover(&out, &errb, []string{"OFF_TRUNK", "--execute", "--trunk", "main"}); rc != 0 {
		t.Fatalf("rc = %d, stderr=%s stdout=%s", rc, errb.String(), out.String())
	}
	want := [][]string{
		{"git", "fetch", "origin", "main"},
		{"git", "merge", "--no-edit", "origin/main"},
	}
	if !reflect.DeepEqual(ran, want) {
		t.Fatalf("ran = %v, want %v", ran, want)
	}
}

func TestRecoverMergeInProgressExecuteRestoresStaged(t *testing.T) {
	old := recoverRunStep
	t.Cleanup(func() { recoverRunStep = old })
	var ran [][]string
	recoverRunStep = func(dir string, argv []string, stdout, stderr io.Writer) int {
		ran = append(ran, append([]string(nil), argv...))
		return 0
	}

	var out, errb bytes.Buffer
	if rc := runRecover(&out, &errb, []string{"MERGE_IN_PROGRESS", "--execute"}); rc != 0 {
		t.Fatalf("rc = %d, stderr=%s stdout=%s", rc, errb.String(), out.String())
	}
	want := [][]string{{"git", "restore", "--staged"}}
	if !reflect.DeepEqual(ran, want) {
		t.Fatalf("ran = %v, want %v", ran, want)
	}
}

func TestRecoverManualPlanRefusesExecute(t *testing.T) {
	var out, errb bytes.Buffer
	if rc := runRecover(&out, &errb, []string{"STALE_RECALL", "--execute"}); rc != 3 {
		t.Fatalf("rc = %d, want 3; stdout=%s stderr=%s", rc, out.String(), errb.String())
	}
	if !strings.Contains(errb.String(), "no safe executable recovery") {
		t.Fatalf("stderr missing refusal: %s", errb.String())
	}
}

func TestRecoverSystemCommitHeadroomIsBoundedAndManual(t *testing.T) {
	var out, errb bytes.Buffer
	if rc := runRecover(&out, &errb, []string{"SYSTEM_COMMIT_HEADROOM", "--dry-run"}); rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, errb.String())
	}
	for _, want := range []string{"recover SYSTEM_COMMIT_HEADROOM", "in-flight managed worker", "sanctioned fleet node", "do not lower FAK_SYSTEM_COMMIT_HEADROOM_MB"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("plan missing %q:\n%s", want, out.String())
		}
	}
	out.Reset()
	errB := &errb
	errB.Reset()
	if rc := runRecover(&out, errB, []string{"SYSTEM_COMMIT_HEADROOM", "--execute"}); rc != 3 {
		t.Fatalf("execute rc=%d want 3; stdout=%s stderr=%s", rc, out.String(), errB.String())
	}
}

func TestRecoverDisambiguationTimeoutRoutesToBoundedWorkerLandFlag(t *testing.T) {
	var out, errb bytes.Buffer
	if rc := runRecover(&out, &errb, []string{"DISAMBIGUATION_TIMEOUT", "--dry-run"}); rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, errb.String())
	}
	for _, want := range []string{
		"recover DISAMBIGUATION_TIMEOUT",
		"fak worktree worker land <same-args> --disambiguation-timeout-ms <milliseconds>",
		"inclusive range 1..900000",
		"120000 ms default",
		"same three witnesses",
		"does not skip, weaken, or retry",
		"before trunk CAS",
	} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("plan missing %q:\n%s", want, out.String())
		}
	}
	out.Reset()
	errb.Reset()
	if rc := runRecover(&out, &errb, []string{"DISAMBIGUATION_TIMEOUT", "--execute"}); rc != 3 {
		t.Fatalf("execute rc=%d want 3; stdout=%s stderr=%s", rc, out.String(), errb.String())
	}
}

// TestRecoverStaleUntrackedRoutesToTheContentComparison binds the STALE_UNTRACKED refusal
// (#5408) to an actionable playbook. The two things the printed plan must carry are the ones
// the refusal itself was written to correct: compare the trunk copy content-to-content rather
// than by diff, and there is a deliberate one-shot escape for a supersede you actually mean.
func TestRecoverStaleUntrackedRoutesToTheContentComparison(t *testing.T) {
	var out, errb bytes.Buffer
	if rc := runRecover(&out, &errb, []string{"STALE_UNTRACKED", "--dry-run", "--trunk", "main"}); rc != 0 {
		t.Fatalf("rc = %d, stderr=%s", rc, errb.String())
	}
	got := out.String()
	for _, want := range []string{
		"recover STALE_UNTRACKED",
		"git fetch origin main",
		"git show origin/main:<path>",
		"FAK_STALE_BASE_GUARD=warn",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("dry-run output missing %q:\n%s", want, got)
		}
	}
}

// TestRecoverStaleUntrackedIsNotAutoExecutable: merging while the path is still untracked can
// stop on an overwrite refusal, so this playbook is read-only by design and --execute must say
// so rather than run half of it.
func TestRecoverStaleUntrackedIsNotAutoExecutable(t *testing.T) {
	var out, errb bytes.Buffer
	if rc := runRecover(&out, &errb, []string{"STALE_UNTRACKED", "--execute"}); rc != 3 {
		t.Fatalf("rc = %d, want 3; stdout=%s stderr=%s", rc, out.String(), errb.String())
	}
	if !strings.Contains(errb.String(), "no safe executable recovery") {
		t.Fatalf("stderr missing refusal: %s", errb.String())
	}
}

// TestRecoverCollisionRiskOffersARouteThatPreservesFinishedWork binds the COLLISION_RISK
// playbook (#5481) to the case its original two routes both dropped. "wait for the live lease"
// and "choose a disjoint lane/region" each assume the change is not written yet; when it is
// already written, built and green, waiting leaves it dirty in a shared checkout (exactly the
// hazard `fak wip sweep-guard` exists to warn about) and it cannot be re-aimed at another lane.
// The plan must therefore name a route that PRESERVES the finished delta — the checkpoint verb —
// and it must be offered first, because it is the only one of the three that is safe to do
// immediately and loses nothing if the operator then also waits or repartitions.
func TestRecoverCollisionRiskOffersARouteThatPreservesFinishedWork(t *testing.T) {
	var out, errb bytes.Buffer
	if rc := runRecover(&out, &errb, []string{"COLLISION_RISK", "--dry-run"}); rc != 0 {
		t.Fatalf("rc = %d, stderr=%s", rc, errb.String())
	}
	got := out.String()
	for _, want := range []string{
		"recover COLLISION_RISK",
		"fak wip checkpoint",
		"dos top",
		"dos arbitrate",
		"fak wip sweep-guard",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("dry-run output missing %q:\n%s", want, got)
		}
	}
	if i, j := strings.Index(got, "fak wip checkpoint"), strings.Index(got, "dos top"); i > j {
		t.Fatalf("checkpoint route must be offered before the wait route (%d > %d):\n%s", i, j, got)
	}

	plan, ok := recoveryPlans("main")["COLLISION_RISK"]
	if !ok {
		t.Fatal("COLLISION_RISK plan missing")
	}
	if len(plan.Steps) == 0 || !reflect.DeepEqual(plan.Steps[0].Argv, []string{"fak", "wip", "checkpoint"}) {
		t.Fatalf("first step = %+v, want the preserve-the-work checkpoint route", plan.Steps)
	}
	if plan.Steps[0].Safe {
		t.Fatal("the checkpoint step must not be marked Safe: --execute would then run it on every collision, which is a behaviour change, not a routing fix")
	}
}

// TestRecoverCollisionRiskStaysManual: adding a concrete command to the plan must not silently
// turn it into an auto-run playbook — a recovery that checkpoints on every collision is a
// behaviour change. --execute still refuses and points at the dry-run notes.
func TestRecoverCollisionRiskStaysManual(t *testing.T) {
	var out, errb bytes.Buffer
	if rc := runRecover(&out, &errb, []string{"COLLISION_RISK", "--execute"}); rc != 3 {
		t.Fatalf("rc = %d, want 3; stdout=%s stderr=%s", rc, out.String(), errb.String())
	}
	if !strings.Contains(errb.String(), "no safe executable recovery") {
		t.Fatalf("stderr missing refusal: %s", errb.String())
	}
}

func TestRecoverUnknownFailsClosed(t *testing.T) {
	var out, errb bytes.Buffer
	if rc := runRecover(&out, &errb, []string{"NOT_A_REASON"}); rc != 2 {
		t.Fatalf("rc = %d, want 2; stdout=%s stderr=%s", rc, out.String(), errb.String())
	}
	if !strings.Contains(errb.String(), "unknown recovery reason") {
		t.Fatalf("stderr = %s", errb.String())
	}
	if !strings.Contains(errb.String(), "NOT_A_REASON") {
		t.Fatalf("stderr should name the refused token: %s", errb.String())
	}
}

func TestRecoveryCatalogCoversEmittedReasons(t *testing.T) {
	plans := recoveryPlans(t.TempDir())
	if !sort.StringsAreSorted(emittedRecoveryReasons) {
		t.Fatalf("emitted recovery vocabulary must stay sorted: %v", emittedRecoveryReasons)
	}
	for _, reason := range emittedRecoveryReasons {
		plan, ok := plans[reason]
		if !ok {
			t.Errorf("emitted refusal %s has no recovery plan", reason)
			continue
		}
		if plan.Reason != reason || plan.Summary == "" || (len(plan.Steps) == 0 && len(plan.Notes) == 0) {
			t.Errorf("emitted refusal %s has incomplete recovery: %+v", reason, plan)
		}
	}
}

func TestRecoverResolvesFrequentLiveRefusals(t *testing.T) {
	for _, reason := range emittedRecoveryReasons {
		t.Run(reason, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			if code := runRecover(&stdout, &stderr, []string{reason}); code != 0 {
				t.Fatalf("runRecover(%s)=%d stderr=%s", reason, code, stderr.String())
			}
			if !strings.Contains(stdout.String(), "recover "+reason+" (dry-run)") {
				t.Fatalf("missing actionable plan header:\n%s", stdout.String())
			}
		})
	}
}

func TestRecoverBehindFastForwardableIsExecutable(t *testing.T) {
	old := recoverRunStep
	t.Cleanup(func() { recoverRunStep = old })
	var ran [][]string
	recoverRunStep = func(dir string, argv []string, stdout, stderr io.Writer) int {
		ran = append(ran, append([]string(nil), argv...))
		return 0
	}

	var out, errb bytes.Buffer
	if rc := runRecover(&out, &errb, []string{"behind-fast-forwardable", "--execute", "--trunk", "main"}); rc != 0 {
		t.Fatalf("rc = %d, stderr=%s stdout=%s", rc, errb.String(), out.String())
	}
	want := [][]string{
		{"fak", "sync", "apply", "--fetch", "--remote", "origin", "--branch", "main"},
	}
	if !reflect.DeepEqual(ran, want) {
		t.Fatalf("ran = %v, want %v", ran, want)
	}
}

func TestRecoverTargetMovedIsExecutable(t *testing.T) {
	old := recoverRunStep
	t.Cleanup(func() { recoverRunStep = old })
	var ran [][]string
	recoverRunStep = func(dir string, argv []string, stdout, stderr io.Writer) int {
		ran = append(ran, append([]string(nil), argv...))
		return 0
	}

	var out, errb bytes.Buffer
	if rc := runRecover(&out, &errb, []string{"target-moved", "--execute", "--trunk", "main"}); rc != 0 {
		t.Fatalf("rc = %d, stderr=%s stdout=%s", rc, errb.String(), out.String())
	}
	want := [][]string{
		{"fak", "sync", "check", "--fetch", "--remote", "origin", "--branch", "main"},
	}
	if !reflect.DeepEqual(ran, want) {
		t.Fatalf("ran = %v, want %v", ran, want)
	}
}

func TestRecoverSyncManualPlansRefuseExecute(t *testing.T) {
	manualTokens := []string{
		"diverged-overlap",
		"diverged-disjoint",
		"merge-active-peer-owned",
		"dirty-write-overlap",
		"queued-awaiting-quiescence",
		"lease-owner-unavailable",
		"behind",
	}
	for _, token := range manualTokens {
		t.Run(token, func(t *testing.T) {
			var out, errb bytes.Buffer
			if rc := runRecover(&out, &errb, []string{token, "--execute"}); rc != 3 {
				t.Fatalf("rc = %d, want 3; stdout=%s stderr=%s", rc, out.String(), errb.String())
			}
			if !strings.Contains(errb.String(), "no safe executable recovery") {
				t.Fatalf("stderr missing refusal: %s", errb.String())
			}
		})
	}
}

func TestRecoveryCatalogCoversABIKernelReasons(t *testing.T) {
	plans := recoveryPlans("main")
	for _, reason := range abi.ReasonNames() {
		plan, ok := plans[reason]
		if !ok {
			t.Errorf("kernel ABI reason %q has no registered recovery plan in fak recover", reason)
			continue
		}
		if plan.Summary == "" {
			t.Errorf("kernel ABI recovery plan for %q has empty Summary", reason)
		}
		if !plan.Executable && len(plan.Notes) == 0 && len(plan.Steps) == 0 {
			t.Errorf("manual recovery plan for %q has neither Notes nor Steps", reason)
		}
	}
}

func TestRecoveryCatalogCoversDOSTokens(t *testing.T) {
	plans := recoveryPlans("main")
	root := guardFindReasonRoot()
	if root == "" {
		t.Fatal("could not find repository root containing dos.toml")
	}
	dosDocs := guardReadReasonDocs(root)
	if len(dosDocs) == 0 {
		t.Fatal("no reasons parsed from dos.toml")
	}
	for token := range dosDocs {
		plan, ok := plans[token]
		if !ok {
			t.Errorf("dos.toml refusal reason %q has no registered recovery plan in fak recover", token)
			continue
		}
		if plan.Summary == "" {
			t.Errorf("recovery plan for dos.toml reason %q has empty Summary", token)
		}
		if len(plan.Notes) == 0 && len(plan.Steps) == 0 {
			t.Errorf("recovery plan for dos.toml reason %q has neither Notes nor Steps", token)
		}
	}
}

func TestRecoverCommittedRedUsesFakDev(t *testing.T) {
	plan, ok := recoveryPlans("main")["COMMITTED_RED"]
	if !ok {
		t.Fatal("COMMITTED_RED plan missing")
	}
	if len(plan.Steps) == 0 || !reflect.DeepEqual(plan.Steps[0].Argv, []string{"fak-dev", "ci-preflight"}) {
		t.Fatalf("COMMITTED_RED step = %+v, want fak-dev ci-preflight", plan.Steps)
	}
}

func TestRecoverConceptFreshnessUsesGenerateStagedFlag(t *testing.T) {
	plan, ok := recoveryPlans("main")["CONCEPT_FRESHNESS"]
	if !ok {
		t.Fatal("CONCEPT_FRESHNESS plan missing")
	}
	if len(plan.Steps) == 0 || !reflect.DeepEqual(plan.Steps[0].Argv, []string{"fak", "concept", "generate", "--staged"}) {
		t.Fatalf("CONCEPT_FRESHNESS step = %+v, want fak concept generate --staged", plan.Steps)
	}
}

func TestRecoverStaleRecallUsesRunIDPlaceholder(t *testing.T) {
	plan, ok := recoveryPlans("main")["STALE_RECALL"]
	if !ok {
		t.Fatal("STALE_RECALL plan missing")
	}
	if len(plan.Steps) == 0 || !reflect.DeepEqual(plan.Steps[0].Argv, []string{"dos", "status", "<run-id>"}) {
		t.Fatalf("STALE_RECALL step = %+v, want dos status <run-id>", plan.Steps)
	}
}

func TestRecoverPolicyBlockUsesPreflight(t *testing.T) {
	plan, ok := recoveryPlans("main")["POLICY_BLOCK"]
	if !ok {
		t.Fatal("POLICY_BLOCK plan missing")
	}
	if len(plan.Steps) == 0 || !reflect.DeepEqual(plan.Steps[0].Argv, []string{"fak", "preflight"}) {
		t.Fatalf("POLICY_BLOCK step = %+v, want fak preflight", plan.Steps)
	}
}

func TestRecoverTrustViolationUsesPreflight(t *testing.T) {
	plan, ok := recoveryPlans("main")["TRUST_VIOLATION"]
	if !ok {
		t.Fatal("TRUST_VIOLATION plan missing")
	}
	if len(plan.Steps) == 0 || !reflect.DeepEqual(plan.Steps[0].Argv, []string{"fak", "preflight"}) {
		t.Fatalf("TRUST_VIOLATION step = %+v, want fak preflight", plan.Steps)
	}
}

func TestRecoverPolicyBlockRecommendsScopedAbstain(t *testing.T) {
	var out, errb bytes.Buffer
	if rc := runRecover(&out, &errb, []string{"POLICY_BLOCK", "--dry-run"}); rc != 0 {
		t.Fatalf("rc = %d, stderr=%s", rc, errb.String())
	}
	got := out.String()
	if !strings.Contains(got, "scoped fail-to-abstain") && !strings.Contains(got, "ABSTAIN") {
		t.Fatalf("expected scoped fail-to-abstain guidance in POLICY_BLOCK output:\n%s", got)
	}
	if !strings.Contains(got, "Operator only — do not attempt from autonomous agent") {
		t.Fatalf("expected operator-only qualification for manual guard overrides in POLICY_BLOCK output:\n%s", got)
	}
}
