package selfinstall

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// hotHost lays out a synthetic host: <root>/fak, <root>/tools/.bin/fak, <home>/bin/fak,
// <home>/go/bin/fak and a scheduled binary, creating exactly the ones named. It returns the
// Host plus the absolute path of each created file, so a test can assert on paths without
// restating the layout rule (which is what Roles owns).
func hotHost(t *testing.T, create ...Role) (Host, map[Role]string) {
	t.Helper()
	root := t.TempDir()
	home := t.TempDir()
	exe := "fak" + binExt()
	paths := map[Role]string{
		RoleGate:      filepath.Join(root, exe),
		RoleWorker:    filepath.Join(root, "tools", ".bin", exe),
		RolePath:      filepath.Join(home, "bin", exe),
		RoleGoBin:     filepath.Join(home, "go", "bin", exe),
		RoleScheduled: filepath.Join(home, "sched", exe),
	}
	for _, r := range create {
		p := paths[r]
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(string(r)), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	return Host{RepoRoot: root, Home: home, Scheduled: paths[RoleScheduled]}, paths
}

// TestRolesDeclaresOneCanonicalPathPerRole is Done condition 1 of #6508: every role has exactly
// one declared path, derived from the repo root / home / scheduled binary and NOTHING else. The
// point is the absence of ambient resolution — a changed PATH must not be able to move a role.
func TestRolesDeclaresOneCanonicalPathPerRole(t *testing.T) {
	h, paths := hotHost(t, RoleGate, RoleWorker)
	got := Roles(h)
	if len(got) != 5 {
		t.Fatalf("Roles() returned %d rows, want one per role: %+v", len(got), got)
	}
	seenRole := map[Role]bool{}
	for _, c := range got {
		if seenRole[c.Role] {
			t.Errorf("role %s appears twice; a role is ONE canonical binary: %+v", c.Role, got)
		}
		seenRole[c.Role] = true
		if want := paths[c.Role]; !strings.EqualFold(c.Path, want) {
			t.Errorf("role %s path = %q, want %q", c.Role, c.Path, want)
		}
	}
	for _, c := range got {
		wantPresent := c.Role == RoleGate || c.Role == RoleWorker
		if c.Present != wantPresent {
			t.Errorf("role %s Present = %v, want %v (only the created files exist)", c.Role, c.Present, wantPresent)
		}
	}
	// A host with no resolvable home / scheduled binary yields FEWER roles, never invented ones.
	if bare := Roles(Host{RepoRoot: h.RepoRoot}); len(bare) != 2 {
		t.Errorf("Roles with no home/scheduled = %d rows, want just gate+worker: %+v", len(bare), bare)
	}
}

// TestCensusOfDivergentCopiesIsNotConverged is the #6508 regression witness in the census layer:
// four deployed binaries on THREE builds, one of them an unreviewed `+uncommitted` compile —
// exactly the host state the issue reported — must audit as NOT CONVERGED, naming every copy.
// Before this, `self-update --target X` converged X and reported success, so this host was
// indistinguishable from a converged one.
func TestCensusOfDivergentCopiesIsNotConverged(t *testing.T) {
	h, paths := hotHost(t, RoleGate, RoleWorker, RolePath, RoleGoBin)
	const head = "0c96937b61ac2e1e9d1f0b3c4d5e6f7a8b9c0d1e"
	builds := map[string]struct {
		build string
		dirty bool
	}{
		strings.ToLower(paths[RoleGate]):   {"e5fc01af20cd0000000000000000000000000000", true},
		strings.ToLower(paths[RoleWorker]): {head, false},
		strings.ToLower(paths[RolePath]):   {head, false},
		strings.ToLower(paths[RoleGoBin]):  {"7298f8f2abbb0000000000000000000000000000", false},
	}
	probe := func(p string) (string, bool, bool) {
		row, ok := builds[strings.ToLower(p)]
		if !ok {
			return "", false, false
		}
		return row.build, row.dirty, true
	}

	audit := AuditCopies(Census(h, probe), head)
	if audit.Converged {
		t.Fatalf("a host on 3 builds with a +uncommitted gate audited as CONVERGED: %+v", audit)
	}
	if got := roleList(audit.Divergent); got != "gate,gobin" {
		t.Errorf("divergent roles = %q, want gate,gobin", got)
	}
	if got := roleList(audit.Dirty); got != "gate" {
		t.Errorf("dirty roles = %q, want gate", got)
	}
	if got := roleList(audit.Missing); got != "scheduled" {
		t.Errorf("missing roles = %q, want scheduled", got)
	}
	if len(audit.Builds) != 3 {
		t.Errorf("distinct builds = %v, want 3", audit.Builds)
	}
	// Every configured hot copy has to be NAMED, including the one self-update may not swap.
	lines := strings.Join(audit.Lines(), "\n")
	for _, role := range []Role{RoleGate, RoleWorker, RolePath, RoleGoBin, RoleScheduled} {
		if !strings.Contains(lines, "role="+string(role)) {
			t.Errorf("audit lines omit role %s — an unnamed hot copy is the silent divergence:\n%s", role, lines)
		}
	}
	if !strings.Contains(lines, "NOT CONVERGED") {
		t.Errorf("audit verdict line does not say NOT CONVERGED:\n%s", lines)
	}
	if !strings.Contains(lines, "DIRTY") {
		t.Errorf("audit lines do not flag the +uncommitted gate build:\n%s", lines)
	}

	// The converged case is the same call with one build everywhere.
	same := func(string) (string, bool, bool) { return head, false, true }
	if ok := AuditCopies(Census(h, same), head); !ok.Converged {
		t.Errorf("four copies on ONE clean build audited as not converged: %+v", ok)
	}
}

// TestAuditFlagsAnUnattestableCopy — a binary that cannot say which commit it is must never
// count as agreeing. Silently treating "unknown" as "the reference build" is how an
// un-attestable copy certifies itself.
func TestAuditFlagsAnUnattestableCopy(t *testing.T) {
	h, paths := hotHost(t, RoleWorker, RolePath)
	const head = "0c96937b61ac2e1e9d1f0b3c4d5e6f7a8b9c0d1e"
	probe := func(p string) (string, bool, bool) {
		if strings.EqualFold(p, paths[RolePath]) {
			return "", false, false // would not run / printed no stamp
		}
		return head, false, true
	}
	audit := AuditCopies(Census(h, probe), head)
	if audit.Converged {
		t.Fatalf("an unattestable copy audited as converged: %+v", audit)
	}
	if got := roleList(audit.Unattested); got != "path" {
		t.Fatalf("unattested roles = %q, want path", got)
	}
	if !strings.Contains(strings.Join(audit.Lines(), "\n"), "UNATTESTED") {
		t.Errorf("audit lines do not name the unattestable copy: %v", audit.Lines())
	}
}

func TestAuditPartitionPreservesStrictVerdictAndRepairScope(t *testing.T) {
	const head = "0c96937b61ac2e1e9d1f0b3c4d5e6f7a8b9c0d1e"
	audit := AuditCopies([]HotCopy{
		{Role: RoleGate, Path: "/repo/fak", Present: true, Attested: true, Dirty: true, Build: "111111111111"},
		{Role: RoleWorker, Path: "/repo/tools/.bin/fak", Present: true, Attested: true, Build: "222222222222"},
		{Role: RolePath, Path: "/home/bin/fak", Present: true, Attested: false},
		{Role: RoleGoBin, Path: "/home/go/bin/fak", Present: true, Attested: true, Build: head},
	}, head)
	if audit.Converged {
		t.Fatal("mixed unsafe roles must keep the strict all-role audit unconverged")
	}
	p := audit.Partition()
	if got := roleList(p.AuditOnly.Dirty); got != "gate" {
		t.Fatalf("audit-only dirty roles = %q, want gate", got)
	}
	if got := roleList(p.AuditOnly.Divergent); got != "gate" {
		t.Fatalf("audit-only divergent roles = %q, want gate", got)
	}
	if got := roleList(p.Convergeable.Divergent); got != "worker" {
		t.Fatalf("convergeable divergent roles = %q, want worker", got)
	}
	if got := roleList(p.Convergeable.Unattested); got != "path" {
		t.Fatalf("convergeable unattested roles = %q, want path", got)
	}
	if !p.Convergeable.Present() || !p.AuditOnly.Present() {
		t.Fatalf("partition lost drift: %+v", p)
	}
}

func TestAuditPartitionTreatsMissingRolesAsReportedNotRepairable(t *testing.T) {
	audit := AuditCopies([]HotCopy{
		{Role: RoleGate, Path: "/repo/fak"},
		{Role: RoleWorker, Path: "/repo/tools/.bin/fak"},
	}, "abcdef012345")
	if !audit.Converged {
		t.Fatalf("missing-only audit changed the existing convergence contract: %+v", audit)
	}
	p := audit.Partition()
	if p.Convergeable.Present() || p.AuditOnly.Present() {
		t.Fatalf("missing roles became updater drift: %+v", p)
	}
}

// TestAuditWithoutAReferenceUsesTheMajorityBuild — a caller with no git context can still ask
// "do these agree with each other?", and the answer must not depend on map iteration order.
func TestAuditWithoutAReferenceUsesTheMajorityBuild(t *testing.T) {
	copies := []HotCopy{
		{Role: RoleGate, Path: "/x/fak", Present: true, Attested: true, Build: "aaaaaaaaaaaa"},
		{Role: RoleWorker, Path: "/y/fak", Present: true, Attested: true, Build: "bbbbbbbbbbbb"},
		{Role: RolePath, Path: "/z/fak", Present: true, Attested: true, Build: "bbbbbbbbbbbb"},
	}
	for i := 0; i < 8; i++ { // repeat: a map-order-dependent reference would flake here
		a := AuditCopies(copies, "")
		if a.Want != "bbbbbbbbbbbb" {
			t.Fatalf("inferred reference = %q, want the majority build", a.Want)
		}
		if roleList(a.Divergent) != "gate" {
			t.Fatalf("divergent = %q, want gate", roleList(a.Divergent))
		}
	}
}

// TestSameRevToleratesShortRevisions — `fak version` prints a short rev while git hands back the
// full 40-hex one; treating those as different builds would report every converged host as
// divergent (and vice versa for a 6-char coincidence, which must NOT match).
func TestSameRevToleratesShortRevisions(t *testing.T) {
	const full = "0c96937b61ac2e1e9d1f0b3c4d5e6f7a8b9c0d1e"
	if !sameRev("0c96937b61ac", full) || !sameRev(full, "0C96937B61AC") {
		t.Errorf("a 12-char prefix of %s must compare equal", full)
	}
	if sameRev("0c9693", full) {
		t.Errorf("a 6-char prefix is too short to identify a commit; must not match")
	}
	if sameRev("", full) || sameRev(full, "") {
		t.Errorf("an empty revision must never match")
	}
}

// TestConvergeTargetsNeverSwapsTheGateBinary pins the convergence POLICY: the repo-root gate
// binary is audit-only (a peer's hand-build in a shared dirty checkout, possibly held live),
// while every other present role is a legitimate unattended swap target. The primary --target is
// skipped because it was already installed.
func TestConvergeTargetsNeverSwapsTheGateBinary(t *testing.T) {
	h, paths := hotHost(t, RoleGate, RoleWorker, RolePath, RoleGoBin, RoleScheduled)
	got := ConvergeTargets(Census(h, nil), paths[RolePath])
	for _, p := range got {
		if strings.EqualFold(p, paths[RoleGate]) {
			t.Errorf("ConvergeTargets included the repo-root gate binary %q: %v", paths[RoleGate], got)
		}
		if strings.EqualFold(p, paths[RolePath]) {
			t.Errorf("ConvergeTargets repeated the primary target %q: %v", paths[RolePath], got)
		}
	}
	want := []string{paths[RoleWorker], paths[RoleGoBin], paths[RoleScheduled]}
	if len(got) != len(want) {
		t.Fatalf("ConvergeTargets = %v, want %v", got, want)
	}
	for i := range want {
		if !strings.EqualFold(got[i], want[i]) {
			t.Fatalf("ConvergeTargets[%d] = %q, want %q (role order must be stable)", i, got[i], want[i])
		}
	}
	// A path that does not exist is never a target: we converge binaries, never create new
	// install locations.
	bare, barePaths := hotHost(t, RoleWorker)
	for _, p := range ConvergeTargets(Census(bare, nil), "") {
		if !strings.EqualFold(p, barePaths[RoleWorker]) {
			t.Errorf("ConvergeTargets invented a path for a role with no binary: %q", p)
		}
	}
}

// TestConvergeTargetsDedupesOneFileNamedByTwoRoles — the scheduled task normally executes the
// PATH copy, so both roles point at one file. Swapping it twice in a tick would leave a second
// swap-aside corpse behind on Windows for no gain.
func TestConvergeTargetsDedupesOneFileNamedByTwoRoles(t *testing.T) {
	h, paths := hotHost(t, RolePath)
	h.Scheduled = paths[RolePath]
	got := ConvergeTargets(Census(h, nil), "")
	if len(got) != 1 || !strings.EqualFold(got[0], paths[RolePath]) {
		t.Fatalf("ConvergeTargets = %v, want exactly one entry for the shared file %q", got, paths[RolePath])
	}
}

// TestNeedsConvergeDemandsProofOfFreshness — the swap decision must fail SAFE. Only a present,
// attested, clean copy already on the reference is skipped; "we could not tell" converges,
// because that state is exactly what let a stale fleet binary survive every tick.
func TestNeedsConvergeDemandsProofOfFreshness(t *testing.T) {
	const head = "0c96937b61ac2e1e9d1f0b3c4d5e6f7a8b9c0d1e"
	base := filepath.Join(t.TempDir(), "fak"+binExt())
	mk := func(mut func(*HotCopy)) []HotCopy {
		c := HotCopy{Role: RolePath, Path: base, Present: true, Attested: true, Build: head}
		mut(&c)
		return []HotCopy{c}
	}
	cases := []struct {
		name string
		cop  []HotCopy
		want bool
	}{
		{"on the reference build", mk(func(*HotCopy) {}), false},
		{"short rev of the reference", mk(func(c *HotCopy) { c.Build = head[:12] }), false},
		{"another build", mk(func(c *HotCopy) { c.Build = "7298f8f2abbb" }), true},
		{"dirty", mk(func(c *HotCopy) { c.Dirty = true }), true},
		{"unattested", mk(func(c *HotCopy) { c.Attested, c.Build = false, "" }), true},
		{"absent", mk(func(c *HotCopy) { c.Present = false }), true},
		{"not in the census", nil, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := NeedsConverge(c.cop, base, head); got != c.want {
				t.Errorf("NeedsConverge(%+v) = %v, want %v", c.cop, got, c.want)
			}
		})
	}
}

// TestPinSkewDetectsAnUnreviewedOrMovedScheduledBinary is Done condition 4 of #6508: a scheduled
// task must pin a reviewed provenance and detect skew BEFORE it executes. The build id itself is
// deliberately NOT pinned — convergence advances it every tick — so what is checked is the path
// and whether that path still holds an attestable, non-dirty build.
func TestPinSkewDetectsAnUnreviewedOrMovedScheduledBinary(t *testing.T) {
	pinned := filepath.Join(t.TempDir(), "bin", "fak"+binExt())
	clean := HotCopy{Role: RoleScheduled, Path: pinned, Present: true, Attested: true,
		Build: "0c96937b61ac2e1e9d1f0b3c4d5e6f7a8b9c0d1e"}
	pin := Pin{Path: pinned, Build: "7298f8f2abbb0000000000000000000000000000"}

	// The healthy tick: same path, still attestable, build has legitimately ADVANCED past the
	// registration-time one. That must NOT be skew, or every post-convergence tick would refuse.
	if skew, why := PinSkew(pin, clean); skew {
		t.Fatalf("a converged pinned binary reported skew: %s", why)
	}

	dirty := clean
	dirty.Dirty = true
	if skew, why := PinSkew(pin, dirty); !skew || !strings.Contains(why, "+uncommitted") {
		t.Errorf("a +uncommitted pinned binary must be skew; got skew=%v why=%q", skew, why)
	}
	unattested := clean
	unattested.Attested, unattested.Build = false, ""
	if skew, why := PinSkew(pin, unattested); !skew || !strings.Contains(why, "no VCS stamp") {
		t.Errorf("an unattestable pinned binary must be skew; got skew=%v why=%q", skew, why)
	}
	moved := clean
	moved.Path = filepath.Join(t.TempDir(), "go", "bin", "fak"+binExt())
	if skew, why := PinSkew(pin, moved); !skew || !strings.Contains(why, "not the pinned one") {
		t.Errorf("executing a DIFFERENT binary than the pinned one must be skew; got skew=%v why=%q", skew, why)
	}
	gone := clean
	gone.Present = false
	if skew, why := PinSkew(pin, gone); !skew || !strings.Contains(why, "no longer on disk") {
		t.Errorf("a pinned binary that vanished must be skew; got skew=%v why=%q", skew, why)
	}
	// No pin at all is itself skew: an unpinned task executes whatever its registration happened
	// to resolve, which is how a stale Go-bin copy ended up certifying evidence.
	if skew, why := PinSkew(Pin{}, clean); !skew || !strings.Contains(why, "pinned no binary provenance") {
		t.Errorf("an unpinned scheduled task must report skew; got skew=%v why=%q", skew, why)
	}
}
