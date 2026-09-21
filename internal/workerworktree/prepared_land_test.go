package workerworktree

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type preparedFixture struct {
	root, wt, base, msg, rel string
	binding                  ProspectiveVerificationBinding
}

type preparedObservation struct {
	verify, prospective int
	sha, bytes          string
}

func newPreparedFixture(t *testing.T, rel string) preparedFixture {
	t.Helper()
	root := t.TempDir()
	mustGit(t, root, "init", "-q", "-b", "main")
	mustGit(t, root, "config", "user.email", "prepared@test.invalid")
	mustGit(t, root, "config", "user.name", "Prepared Test")
	mustGit(t, root, "config", "commit.gpgsign", "false")
	if err := os.WriteFile(filepath.Join(root, "base.txt"), []byte("base\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	mustGit(t, root, "add", "base.txt")
	mustGit(t, root, "commit", "-q", "-m", "base")
	base := strings.TrimSpace(mustGit(t, root, "rev-parse", "refs/heads/main"))
	prep := Prepare(root, "prepared-land", strings.ReplaceAll(rel, "/", "-"), base, t.TempDir(), nil)
	if !prep.OK {
		t.Fatalf("prepare checkout: %+v", prep)
	}
	target := filepath.Join(prep.Path, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(target, []byte("candidate bytes\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	msg := filepath.Join(t.TempDir(), "message.txt")
	if err := os.WriteFile(msg, []byte("feat(workerworktree): prepared candidate (fak workerworktree)\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	return preparedFixture{root: root, wt: prep.Path, base: base, msg: msg, rel: rel,
		binding: ProspectiveVerificationBinding{Command: "go test ./...", Tags: []string{"unit", "linux"}}}
}

func prepareCandidate(t *testing.T, f preparedFixture) (PreparedLandReceipt, Result, *preparedObservation) {
	t.Helper()
	obs := &preparedObservation{}
	verify := func(string) (bool, string) { obs.verify++; return true, "verified" }
	prospective := func(dir string, materializationErr error) Result {
		obs.prospective++
		if materializationErr != nil {
			return Result{OK: false, Detail: materializationErr.Error()}
		}
		data, err := os.ReadFile(filepath.Join(dir, filepath.FromSlash(f.rel)))
		if err != nil {
			return Result{OK: false, Detail: err.Error()}
		}
		obs.bytes = string(data)
		obs.sha = strings.TrimSpace(mustGit(t, dir, "rev-parse", "HEAD"))
		return Result{OK: true}
	}
	r, res := PrepareProspectiveLand(f.root, f.wt, f.base, f.msg, []string{f.rel}, f.binding, verify, prospective, nil)
	return r, res, obs
}

func preparedWant(f preparedFixture, r PreparedLandReceipt) PreparedLandExpectation {
	return PreparedLandExpectation{ReceiptID: r.ReceiptID, Paths: []string{f.rel}, Verification: f.binding}
}

func TestPrepareProspectiveLandBindsExactCandidateAndAcceptDoesNotVerify(t *testing.T) {
	f := newPreparedFixture(t, "feature.txt")
	r, prep, obs := prepareCandidate(t, f)
	if !prep.OK || prep.Code != LandResultPrepared || prep.Applied || prep.Committed {
		t.Fatalf("prepare: %+v", prep)
	}
	if got := strings.TrimSpace(mustGit(t, f.root, "rev-parse", "refs/heads/main")); got != f.base {
		t.Fatalf("prepare moved trunk: %s", got)
	}
	if obs.prospective != 1 || obs.sha != r.CandidateSHA || obs.bytes != "candidate bytes\n" {
		t.Fatalf("verifier observation=%+v receipt=%+v", obs, r)
	}
	if r.ParentSHA != f.base || r.TreeSHA == "" || r.PathsDigest == "" || r.VerifyCommandDigest == "" || r.TagsDigest == "" || r.RecoveryRef == "" {
		t.Fatalf("incomplete binding: %+v", r)
	}
	if parent := strings.TrimSpace(mustGit(t, f.root, "rev-parse", r.CandidateSHA+"^")); parent != r.ParentSHA {
		t.Fatalf("candidate parent=%s receipt parent=%s", parent, r.ParentSHA)
	}
	if tree := strings.TrimSpace(mustGit(t, f.root, "rev-parse", r.CandidateSHA+"^{tree}")); tree != r.TreeSHA {
		t.Fatalf("candidate tree=%s receipt tree=%s", tree, r.TreeSHA)
	}
	if got := strings.TrimSpace(mustGit(t, f.root, "rev-parse", r.RecoveryRef)); got != r.CandidateSHA {
		t.Fatalf("recovery=%s candidate=%s", got, r.CandidateSHA)
	}
	v, p := obs.verify, obs.prospective
	got := AcceptPreparedLand(f.root, f.wt, preparedWant(f, r), nil)
	if !got.OK || got.Code != LandResultSuccess || !got.Applied || !got.Committed {
		t.Fatalf("accept: %+v", got)
	}
	if obs.verify != v || obs.prospective != p {
		t.Fatal("accept reran verification")
	}
	if head := strings.TrimSpace(mustGit(t, f.root, "rev-parse", "refs/heads/main")); head != r.CandidateSHA {
		t.Fatalf("head=%s candidate=%s", head, r.CandidateSHA)
	}
}

func TestAcceptPreparedLandRejectsAlteredBindingsWithoutRefMovement(t *testing.T) {
	tests := []struct {
		name  string
		alter func(*testing.T, preparedFixture, PreparedLandReceipt, *PreparedLandExpectation)
	}{
		{"parent", func(t *testing.T, f preparedFixture, r PreparedLandReceipt, _ *PreparedLandExpectation) {
			rewritePrepared(t, f, r, func(v *PreparedLandReceipt) { v.ParentSHA = strings.Repeat("a", 40) })
		}},
		{"tree", func(t *testing.T, f preparedFixture, r PreparedLandReceipt, _ *PreparedLandExpectation) {
			rewritePrepared(t, f, r, func(v *PreparedLandReceipt) { v.TreeSHA = strings.Repeat("b", 40) })
		}},
		{"candidate", func(t *testing.T, f preparedFixture, r PreparedLandReceipt, _ *PreparedLandExpectation) {
			rewritePrepared(t, f, r, func(v *PreparedLandReceipt) { v.CandidateSHA = strings.Repeat("c", 40) })
		}},
		{"paths", func(_ *testing.T, _ preparedFixture, _ PreparedLandReceipt, e *PreparedLandExpectation) {
			e.Paths = []string{"other.txt"}
		}},
		{"command", func(_ *testing.T, _ preparedFixture, _ PreparedLandReceipt, e *PreparedLandExpectation) {
			e.Verification.Command = "go test ./other"
		}},
		{"tags", func(_ *testing.T, _ preparedFixture, _ PreparedLandReceipt, e *PreparedLandExpectation) {
			e.Verification.Tags = []string{"other"}
		}},
		{"contract", func(t *testing.T, f preparedFixture, r PreparedLandReceipt, _ *PreparedLandExpectation) {
			rewritePrepared(t, f, r, func(v *PreparedLandReceipt) { v.GateContract = "other" })
		}},
		{"verdict", func(t *testing.T, f preparedFixture, r PreparedLandReceipt, _ *PreparedLandExpectation) {
			rewritePrepared(t, f, r, func(v *PreparedLandReceipt) { v.Verdict = "FAIL" })
		}},
		{"recovery", func(t *testing.T, f preparedFixture, r PreparedLandReceipt, _ *PreparedLandExpectation) {
			rewritePrepared(t, f, r, func(v *PreparedLandReceipt) { v.RecoveryRef += "-altered" })
		}},
		{"missing recovery", func(t *testing.T, f preparedFixture, r PreparedLandReceipt, _ *PreparedLandExpectation) {
			mustGit(t, f.root, "update-ref", "-d", r.RecoveryRef)
		}},
		{"malformed", func(t *testing.T, f preparedFixture, r PreparedLandReceipt, _ *PreparedLandExpectation) {
			path, err := PreparedLandReceiptPath(f.root, r.ReceiptID, nil)
			if err != nil {
				t.Fatal(err)
			}
			if err = os.WriteFile(path, []byte("{malformed"), 0o644); err != nil {
				t.Fatal(err)
			}
		}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			f := newPreparedFixture(t, "feature.txt")
			r, prep, obs := prepareCandidate(t, f)
			if !prep.OK {
				t.Fatalf("prepare: %+v", prep)
			}
			want := preparedWant(f, r)
			tc.alter(t, f, r, &want)
			before := strings.TrimSpace(mustGit(t, f.root, "rev-parse", "refs/heads/main"))
			v, p := obs.verify, obs.prospective
			got := AcceptPreparedLand(f.root, f.wt, want, nil)
			if got.OK || got.Code != LandResultPreparedMismatch || !got.Preserved {
				t.Fatalf("altered %s accepted: %+v", tc.name, got)
			}
			if after := strings.TrimSpace(mustGit(t, f.root, "rev-parse", "refs/heads/main")); after != before {
				t.Fatalf("%s moved ref %s -> %s", tc.name, before, after)
			}
			if obs.verify != v || obs.prospective != p {
				t.Fatalf("%s reran verification", tc.name)
			}
		})
	}
}

func TestAcceptPreparedLandStaleParentRequiresReprepareWithoutRetry(t *testing.T) {
	f := newPreparedFixture(t, "feature.txt")
	r, prep, obs := prepareCandidate(t, f)
	if !prep.OK {
		t.Fatalf("prepare: %+v", prep)
	}
	if err := os.WriteFile(filepath.Join(f.root, "peer.txt"), []byte("peer\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	mustGit(t, f.root, "add", "peer.txt")
	mustGit(t, f.root, "commit", "-q", "-m", "peer")
	peer := strings.TrimSpace(mustGit(t, f.root, "rev-parse", "refs/heads/main"))
	cas := 0
	git := func(root string, args []string) (int, string) {
		if len(args) > 0 && args[0] == "update-ref" {
			cas++
		}
		return defaultGit(root, args)
	}
	v, p := obs.verify, obs.prospective
	got := AcceptPreparedLand(f.root, f.wt, preparedWant(f, r), git)
	if got.OK || got.Code != LandResultPreparedReprepare || !got.Preserved {
		t.Fatalf("stale result: %+v", got)
	}
	if cas != 0 {
		t.Fatalf("stale accept attempted %d CAS operations", cas)
	}
	if after := strings.TrimSpace(mustGit(t, f.root, "rev-parse", "refs/heads/main")); after != peer {
		t.Fatalf("peer ref moved: %s", after)
	}
	if obs.verify != v || obs.prospective != p {
		t.Fatal("stale accept reran verification")
	}
}

func TestAcceptPreparedLandLostCASRequiresReprepareWithoutRetry(t *testing.T) {
	f := newPreparedFixture(t, "feature.txt")
	r, prep, obs := prepareCandidate(t, f)
	if !prep.OK {
		t.Fatalf("prepare: %+v", prep)
	}

	mustGit(t, f.root, "checkout", "-q", "-b", "peer")
	if err := os.WriteFile(filepath.Join(f.root, "peer.txt"), []byte("peer\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	mustGit(t, f.root, "add", "peer.txt")
	mustGit(t, f.root, "commit", "-q", "-m", "peer")
	peer := strings.TrimSpace(mustGit(t, f.root, "rev-parse", "HEAD"))
	mustGit(t, f.root, "checkout", "-q", "main")

	cas := 0
	git := func(root string, args []string) (int, string) {
		if len(args) >= 4 && args[0] == "update-ref" && args[1] == "refs/heads/main" {
			cas++
			if rc, out := defaultGit(root, []string{"update-ref", "refs/heads/main", peer, f.base}); rc != 0 {
				t.Fatalf("advance peer during CAS window: rc=%d out=%s", rc, out)
			}
		}
		return defaultGit(root, args)
	}
	v, p := obs.verify, obs.prospective
	got := AcceptPreparedLand(f.root, f.wt, preparedWant(f, r), git)
	if got.OK || got.Code != LandResultPreparedReprepare || !got.Preserved {
		t.Fatalf("lost CAS result: %+v", got)
	}
	if cas != 1 {
		t.Fatalf("lost CAS attempted %d target updates, want exactly 1", cas)
	}
	if after := strings.TrimSpace(mustGit(t, f.root, "rev-parse", "refs/heads/main")); after != peer {
		t.Fatalf("peer ref moved: got=%s want=%s", after, peer)
	}
	if obs.verify != v || obs.prospective != p {
		t.Fatal("lost CAS rebuilt or reran verification")
	}
}

func TestPrepareProspectiveLandPreservesCoreLockGuard(t *testing.T) {
	f := newPreparedFixture(t, "internal/adjudicator/prepared_land_probe.go")
	before := strings.TrimSpace(mustGit(t, f.root, "rev-parse", "refs/heads/main"))
	r, got, _ := prepareCandidate(t, f)
	if got.OK || r.ReceiptID != "" {
		t.Fatalf("core-lock path prepared: receipt=%+v result=%+v", r, got)
	}
	if after := strings.TrimSpace(mustGit(t, f.root, "rev-parse", "refs/heads/main")); after != before {
		t.Fatalf("core-lock refusal moved ref")
	}
}

func TestAcceptPreparedLandFreshAdmissionPreservesRootState(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*testing.T, preparedFixture) string
		check  func(*testing.T, preparedFixture)
	}{
		{
			name: "root-owned path dirt",
			mutate: func(t *testing.T, f preparedFixture) string {
				path := filepath.Join(f.root, filepath.FromSlash(f.rel))
				if err := os.WriteFile(path, []byte("root-owned wip\n"), 0o644); err != nil {
					t.Fatal(err)
				}
				return strings.TrimSpace(mustGit(t, f.root, "rev-parse", "refs/heads/main"))
			},
			check: func(t *testing.T, f preparedFixture) {
				got, err := os.ReadFile(filepath.Join(f.root, filepath.FromSlash(f.rel)))
				if err != nil || string(got) != "root-owned wip\n" {
					t.Fatalf("root WIP was not preserved: bytes=%q err=%v", got, err)
				}
			},
		},
		{
			name: "root switched branch",
			mutate: func(t *testing.T, f preparedFixture) string {
				mustGit(t, f.root, "checkout", "-q", "-b", "peer-branch")
				return strings.TrimSpace(mustGit(t, f.root, "rev-parse", "refs/heads/main"))
			},
			check: func(t *testing.T, f preparedFixture) {
				if branch := strings.TrimSpace(mustGit(t, f.root, "branch", "--show-current")); branch != "peer-branch" {
					t.Fatalf("root branch changed: got %q", branch)
				}
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			f := newPreparedFixture(t, "feature.txt")
			r, prep, obs := prepareCandidate(t, f)
			if !prep.OK {
				t.Fatalf("prepare: %+v", prep)
			}
			before := tc.mutate(t, f)
			v, p := obs.verify, obs.prospective
			got := AcceptPreparedLand(f.root, f.wt, preparedWant(f, r), nil)
			if got.OK || !got.Preserved {
				t.Fatalf("fresh admission change accepted: %+v", got)
			}
			if after := strings.TrimSpace(mustGit(t, f.root, "rev-parse", "refs/heads/main")); after != before {
				t.Fatalf("fresh refusal moved main: before=%s after=%s", before, after)
			}
			if obs.verify != v || obs.prospective != p {
				t.Fatal("fresh refusal reran verification")
			}
			tc.check(t, f)
		})
	}
}
func TestPreparedLandRejectsNonContainedPathsBeforeRefMovement(t *testing.T) {
	absolute := filepath.Join(t.TempDir(), "absolute-outside.txt")
	for _, bad := range []string{`..\outside`, "../outside", absolute, `C:foo`, "bad\x00path"} {
		name := strings.ReplaceAll(strings.ReplaceAll(bad, `\`, "_"), "/", "_")
		t.Run("prepare_"+name, func(t *testing.T) {
			f := newPreparedFixture(t, "feature.txt")
			before := strings.TrimSpace(mustGit(t, f.root, "rev-parse", "refs/heads/main"))
			prospective := 0
			r, got := PrepareProspectiveLand(f.root, f.wt, f.base, f.msg, []string{bad}, f.binding,
				func(string) (bool, string) { return true, "verified" },
				func(string, error) Result { prospective++; return Result{OK: true} }, nil)
			if got.OK || r.ReceiptID != "" {
				t.Fatalf("invalid path %q prepared: receipt=%+v result=%+v", bad, r, got)
			}
			if prospective != 0 {
				t.Fatalf("invalid path %q reached prospective verifier", bad)
			}
			if after := strings.TrimSpace(mustGit(t, f.root, "rev-parse", "refs/heads/main")); after != before {
				t.Fatalf("invalid prepare path %q moved ref", bad)
			}
		})
		t.Run("accept_"+name, func(t *testing.T) {
			f := newPreparedFixture(t, "feature.txt")
			r, prep, obs := prepareCandidate(t, f)
			if !prep.OK {
				t.Fatalf("prepare: %+v", prep)
			}
			want := preparedWant(f, r)
			want.Paths = []string{bad}
			before := strings.TrimSpace(mustGit(t, f.root, "rev-parse", "refs/heads/main"))
			v, p := obs.verify, obs.prospective
			got := AcceptPreparedLand(f.root, f.wt, want, nil)
			if got.OK || got.Code != LandResultPreparedMismatch || !got.Preserved {
				t.Fatalf("invalid accept path %q result=%+v", bad, got)
			}
			if after := strings.TrimSpace(mustGit(t, f.root, "rev-parse", "refs/heads/main")); after != before {
				t.Fatalf("invalid accept path %q moved ref", bad)
			}
			if obs.verify != v || obs.prospective != p {
				t.Fatalf("invalid accept path %q reran verification", bad)
			}
		})
	}
}

func TestAcceptPreparedLandSupportsSpacedPathWithDisjointRootWIP(t *testing.T) {
	f := newPreparedFixture(t, "dir/file with spaces.txt")
	r, prep, obs := prepareCandidate(t, f)
	if !prep.OK {
		t.Fatalf("prepare spaced path: %+v", prep)
	}
	disjoint := filepath.Join(f.root, "unrelated-root-wip.txt")
	if err := os.WriteFile(disjoint, []byte("preserve unrelated wip\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	v, p := obs.verify, obs.prospective
	got := AcceptPreparedLand(f.root, f.wt, preparedWant(f, r), nil)
	if !got.OK || !got.Committed {
		t.Fatalf("accept spaced path with disjoint root WIP: %+v", got)
	}
	if obs.verify != v || obs.prospective != p {
		t.Fatal("accept reran verification")
	}
	landed, err := os.ReadFile(filepath.Join(f.root, "dir", "file with spaces.txt"))
	if err != nil || string(landed) != "candidate bytes\n" {
		t.Fatalf("spaced path did not land exactly: bytes=%q err=%v", landed, err)
	}
	wip, err := os.ReadFile(disjoint)
	if err != nil || string(wip) != "preserve unrelated wip\n" {
		t.Fatalf("disjoint root WIP was not preserved: bytes=%q err=%v", wip, err)
	}
}

func TestPrepareProspectiveLandCoreLockRejectsProtectedPathAfterOrdinaryPath(t *testing.T) {
	f := newPreparedFixture(t, "aaa-ordinary.txt")
	protected := "internal/adjudicator/prepared_land_second.go"
	target := filepath.Join(f.wt, filepath.FromSlash(protected))
	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(target, []byte("package adjudicator\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	before := strings.TrimSpace(mustGit(t, f.root, "rev-parse", "refs/heads/main"))
	r, got := PrepareProspectiveLand(f.root, f.wt, f.base, f.msg,
		[]string{f.rel, protected}, f.binding,
		func(string) (bool, string) { return true, "verified" },
		func(string, error) Result { return Result{OK: true} }, nil)
	if got.OK || r.ReceiptID != "" {
		t.Fatalf("multi-path core-lock candidate prepared: receipt=%+v result=%+v", r, got)
	}
	if after := strings.TrimSpace(mustGit(t, f.root, "rev-parse", "refs/heads/main")); after != before {
		t.Fatalf("multi-path core-lock refusal moved ref: before=%s after=%s", before, after)
	}
}

func rewritePrepared(t *testing.T, f preparedFixture, r PreparedLandReceipt, mutate func(*PreparedLandReceipt)) {
	t.Helper()
	path, err := PreparedLandReceiptPath(f.root, r.ReceiptID, nil)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var stored PreparedLandReceipt
	if err = json.Unmarshal(raw, &stored); err != nil {
		t.Fatal(err)
	}
	mutate(&stored)
	raw, err = json.Marshal(stored)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(path, append(raw, '\n'), 0o644); err != nil {
		t.Fatal(err)
	}
}
