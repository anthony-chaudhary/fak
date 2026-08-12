package main

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// cmd/fak/hygiene_push_scope_test.go — the #6013 contract, pinned at the seam the PUSH actually
// runs.
//
// tools/githooks/pre-push calls exactly `fak hygiene --root <root> --gates TIER_DECLARED` and
// refuses the push on exit 1 (pre-push:110-126). TIER_DECLARED is a WHOLE-TREE gate
// (gateTierDeclaredTree), so an internal/<leaf> that some peer already published without a tier
// row is a finding for whoever pushes NEXT — two such leaves (citeverify, usagepreflight) wedged
// every agent's `fak sync push` in #6013, and neither pusher's commit touched either leaf.
//
// The demotion that answers it lives in two places that only compose at this command boundary:
// hooks.ScopeTierDeclaredFindings marks a TIER_DECLARED finding outside the committed push delta
// Advisory, and runHygiene sets `blocked` only from NON-advisory findings (hygiene.go:118-126,
// 163-169). internal/hooks already unit-tests the demotion function itself
// (TestScopeTierDeclaredFindingsBlocksOnlyPushOwnedLeaf); what was untested is the composition —
// that an untouched untiered leaf leaves the EXIT CODE at 0, which is the only thing the push
// hook reads. A regression that re-promoted the finding, or that stopped calling the scoper from
// runHygiene at all, would keep that unit test green and still wedge the fleet.
//
// The pair below pins both directions, because "advisory" must not become "disabled": the leaf's
// OWN author, whose push delivers it, still gets refused, and a checkout that cannot read the
// trunk (no origin/main -> scoped=false) still blocks rather than silently allowing drift.

// seedTierDriftRepo builds a throwaway git repo whose tier table declares internal/owned but NOT
// internal/peerleaf — the shape of a leaf a peer published untiered. Only `git add` is needed:
// TrackedTree reads `git ls-files`, so staged-but-uncommitted files are already "tracked" and the
// fixture needs no commit identity. Paths are staged EXPLICITLY (never `git add -A`), matching the
// repo's own rule.
func seedTierDriftRepo(t *testing.T) string {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	root := t.TempDir()
	files := map[string]string{
		// The tier table's own file is a _test.go, so it never makes architest a leaf itself.
		"internal/architest/architest_test.go": "package architest\n\nvar tier = map[string]int{\n\t\"owned\": 1,\n}\n",
		"internal/owned/x.go":                  "package owned\n",
		"internal/peerleaf/x.go":               "package peerleaf\n",
	}
	var rel []string
	for name, body := range files {
		p := filepath.Join(root, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		rel = append(rel, name)
	}
	for _, args := range [][]string{{"init", "-q"}, append([]string{"add", "--"}, rel...)} {
		cmd := exec.Command("git", args...)
		cmd.Dir = root
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Skipf("git %v in fixture repo: %v\n%s", args, err, out)
		}
	}
	return root
}

// stubPushDelta replaces the committed-push-delta reader for one test. The real one shells out to
// `git merge-base origin/main HEAD` + `git diff`, which a fixture repo with no remote cannot
// answer; the point under test is what runHygiene DOES with each answer.
func stubPushDelta(t *testing.T, paths []string, scoped bool) {
	t.Helper()
	prev := hygienePushDelta
	hygienePushDelta = func(string) ([]string, bool) { return paths, scoped }
	t.Cleanup(func() { hygienePushDelta = prev })
}

func runTierHygiene(t *testing.T, root string) (int, string) {
	t.Helper()
	var out, errb bytes.Buffer
	rc := runHygiene(&out, &errb, []string{"--root", root, "--gates", "TIER_DECLARED"})
	return rc, out.String() + errb.String()
}

// TestRunHygiene_UntouchedUntieredLeafDoesNotWedgePush is #6013's done-condition 3: a tree carries
// an untiered leaf this push does not touch, and the push PROCEEDS (exit 0 — what the pre-push
// hook reads) while the drift stays VISIBLE to its owner.
func TestRunHygiene_UntouchedUntieredLeafDoesNotWedgePush(t *testing.T) {
	root := seedTierDriftRepo(t)
	stubPushDelta(t, []string{"internal/owned/x.go"}, true)

	rc, report := runTierHygiene(t, root)
	if rc != 0 {
		t.Fatalf("runHygiene exited %d — a peer's untiered leaf this push does not touch must NOT "+
			"refuse the push (#6013); report:\n%s", rc, report)
	}
	if !strings.Contains(report, "internal/peerleaf/") {
		t.Errorf("the untiered leaf vanished from the report — demoted must stay VISIBLE so its "+
			"owner still sees it; report:\n%s", report)
	}
	if !strings.Contains(report, "advisory; outside this push") {
		t.Errorf("the finding is not labelled as outside this push; report:\n%s", report)
	}
}

// TestRunHygiene_PushOwnedUntieredLeafStillRefuses is the other direction: demoting a peer's leaf
// must not disable the gate. When the push delta DELIVERS the untiered leaf, its author is still
// refused at their own push — the refusal lands on the one person who can pick the tier.
func TestRunHygiene_PushOwnedUntieredLeafStillRefuses(t *testing.T) {
	root := seedTierDriftRepo(t)
	stubPushDelta(t, []string{"internal/peerleaf/x.go"}, true)

	rc, report := runTierHygiene(t, root)
	if rc != 1 {
		t.Fatalf("runHygiene exited %d, want 1 — a leaf THIS push delivers untiered must still "+
			"refuse; report:\n%s", rc, report)
	}
	if strings.Contains(report, "advisory; outside this push") {
		t.Errorf("the push's own leaf was demoted to advisory; report:\n%s", report)
	}
}

// TestRunHygiene_TierDriftRefusesWhenPushScopeUnreadable pins the safe fallback: with no trunk to
// diff against (fresh clone, archive checkout, no origin/main) the scoper cannot tell whose leaf
// it is, so every finding stays blocking. Fail-CLOSED here is deliberate — the alternative is a
// checkout where an unreadable remote silently turns the gate off.
func TestRunHygiene_TierDriftRefusesWhenPushScopeUnreadable(t *testing.T) {
	root := seedTierDriftRepo(t)
	stubPushDelta(t, nil, false)

	rc, report := runTierHygiene(t, root)
	if rc != 1 {
		t.Fatalf("runHygiene exited %d, want 1 — with no readable push delta the gate must stay "+
			"blocking; report:\n%s", rc, report)
	}
}
