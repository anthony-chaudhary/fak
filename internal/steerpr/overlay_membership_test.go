package steerpr

// Membership QA (#5033): proves the overlay fold's partition over
// (units, unstamped) is TOTAL and DISJOINT — every commit in a range lands in
// exactly one of {a unit, the orphan/unstamped set}; none in zero, none in two —
// and that orphan (unstamped) commits are DETECTED and surfaced as legibility
// debt with their SHAs, never dropped and never treated as an error.
//
// The adversarial table pins the real stamp-grammar behavior (end-anchored
// regex) rather than changing it: a grammar change is a different ticket.
//
// Witness gate: go test ./internal/steerpr/... -run Membership -count=1

import (
	"fmt"
	"math/rand"
	"strings"
	"testing"
)

// membershipRecord builds one record in the `git log --no-merges --name-only
// --format=%x1e%H%x1f%s%x1f%b%x1f` wire format the real fold consumes. It is a
// deliberate local duplicate of the sibling test helper so this file stays
// self-contained.
func membershipRecord(sha, subject, body string, files ...string) string {
	return "\x1e" + sha + "\x1f" + subject + "\x1f" + body + "\x1f" + strings.Join(files, "\n")
}

// membershipPartition folds commits and returns (sha -> unit leaf) for members,
// (sha -> true) for orphans, and how many times each SHA was seen anywhere in
// the partition. It is the one place the totality + disjointness arithmetic is
// computed, so every Membership test asserts the same invariant the same way.
func membershipPartition(t *testing.T, commits []Commit) (memberLeaf map[string]string, orphan map[string]bool, seen map[string]int, units []Unit, unstamped []Commit) {
	t.Helper()
	units, unstamped = FoldUnits(commits)

	memberLeaf = map[string]string{}
	orphan = map[string]bool{}
	seen = map[string]int{}
	members := 0
	for _, u := range units {
		members += len(u.Commits)
		for _, c := range u.Commits {
			seen[c.SHA]++
			memberLeaf[c.SHA] = u.Leaf
		}
	}
	for _, c := range unstamped {
		seen[c.SHA]++
		orphan[c.SHA] = true
	}

	// The arithmetic invariant the overlay's credibility rests on:
	// commits_seen == sum(unit members) + orphans.
	if members+len(unstamped) != len(commits) {
		t.Errorf("partition not total: members(%d) + orphans(%d) != commits_seen(%d)",
			members, len(unstamped), len(commits))
	}
	// Totality + disjointness per commit: in the partition exactly once.
	for _, c := range commits {
		if seen[c.SHA] != 1 {
			t.Errorf("commit %s appears %d times across the partition, want exactly 1 (total + disjoint)",
				c.SHA, seen[c.SHA])
		}
	}
	return memberLeaf, orphan, seen, units, unstamped
}

// TestMembershipAdversarialPartitionIsTotalAndDisjoint drives the full
// adversarial corpus through the real ParseLog -> FoldUnits pipeline and
// asserts, per case, the EXPECTED partition — which unit a commit lands in, or
// that it is detected as an orphan — not merely "does not panic".
func TestMembershipAdversarialPartitionIsTotalAndDisjoint(t *testing.T) {
	cases := []struct {
		name    string
		sha     string
		subject string
		body    string
		// wantLeaf is the unit the commit must land in; "" means it must be
		// detected as an orphan (unstamped) instead.
		wantLeaf string
	}{
		{
			name: "well-formed stamp", sha: "m1",
			subject:  "feat(steerpr): add the fold (fak steerpr)",
			wantLeaf: "steerpr",
		},
		{
			name: "no stamp at all", sha: "m2",
			subject:  "chore: tidy things with no stamp",
			wantLeaf: "",
		},
		{
			name: "malformed stamp: uppercase leaf", sha: "m3",
			subject:  "fix(cache): warm the ladder (fak Cache)",
			wantLeaf: "",
		},
		{
			name: "malformed stamp: empty leaf", sha: "m4",
			subject:  "fix(cache): warm the ladder (fak )",
			wantLeaf: "",
		},
		{
			name: "malformed stamp: leading hyphen leaf", sha: "m5",
			subject:  "fix(cache): warm the ladder (fak -cache)",
			wantLeaf: "",
		},
		{
			// The interesting case: the grammar anchors to end-of-subject, so
			// with two stamps the FINAL one owns the commit and the mid-subject
			// one is ignored. Pinned deliberately — the commit must land in
			// exactly ONE unit, never both.
			name: "two stamps in one subject: final wins", sha: "m6",
			subject:  "fix(relay): rewire the seam (fak first) (fak second)",
			wantLeaf: "second",
		},
		{
			name: "stamp mid-subject only is not a stamp", sha: "m7",
			subject:  "fix(relay): stamp is (fak mid) not at the end",
			wantLeaf: "",
		},
		{
			name: "stamp in body but not subject is not a stamp", sha: "m8",
			subject:  "fix(relay): forgot the subject stamp",
			body:     "should have been stamped\n(fak relay)\n",
			wantLeaf: "",
		},
		{
			// The fold is normally fed --no-merges output, but a merge subject
			// that slips through must still partition (as an orphan), never
			// vanish from the view.
			name: "merge commit subject", sha: "m9",
			subject:  "Merge branch 'wip' into main",
			wantLeaf: "",
		},
		{
			name: "subject that is only a stamp", sha: "m10",
			subject:  "(fak steerpr)",
			wantLeaf: "steerpr",
		},
	}

	var records []string
	for _, tc := range cases {
		records = append(records, membershipRecord(tc.sha, tc.subject, tc.body, "some/file.go"))
	}
	commits := ParseLog(strings.Join(records, ""))
	if len(commits) != len(cases) {
		t.Fatalf("ParseLog() = %d commits, want %d (every adversarial case must survive parsing)",
			len(commits), len(cases))
	}

	memberLeaf, orphan, _, _, _ := membershipPartition(t, commits)

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if tc.wantLeaf == "" {
				if !orphan[tc.sha] {
					t.Errorf("commit %s (%q): want orphan, got unit %q", tc.sha, tc.subject, memberLeaf[tc.sha])
				}
				return
			}
			if orphan[tc.sha] {
				t.Errorf("commit %s (%q): want unit %q, got orphan", tc.sha, tc.subject, tc.wantLeaf)
				return
			}
			if memberLeaf[tc.sha] != tc.wantLeaf {
				t.Errorf("commit %s (%q): in unit %q, want %q", tc.sha, tc.subject, memberLeaf[tc.sha], tc.wantLeaf)
			}
		})
	}
}

// TestMembershipEmptyRange proves the base case: an empty range partitions to
// zero units and zero orphans, and the arithmetic invariant holds at 0 == 0+0.
func TestMembershipEmptyRange(t *testing.T) {
	commits := ParseLog("")
	if len(commits) != 0 {
		t.Fatalf("ParseLog(\"\") = %d commits, want 0", len(commits))
	}
	_, _, _, units, unstamped := membershipPartition(t, commits)
	if len(units) != 0 || len(unstamped) != 0 {
		t.Errorf("empty range: units=%d unstamped=%d, want 0 and 0", len(units), len(unstamped))
	}
}

// TestMembershipOrphansAreSurfacedAsDebtNotDropped proves the orphan detector's
// contract: an unstamped commit is DEBT, not an error — it comes back with its
// SHA and subject intact so the view can report it, and an ungraded orphan is
// banded UNVERIFIABLE (never laundered to CLEARED, never refused).
func TestMembershipOrphansAreSurfacedAsDebtNotDropped(t *testing.T) {
	raw := membershipRecord("o1", "chore: an unstamped landing", "", "misc.txt") +
		membershipRecord("s1", "feat(steerpr): a stamped landing (fak steerpr)", "", "a.go")
	commits := ParseLog(raw)
	units, unstamped := FoldUnits(commits)

	if len(unstamped) != 1 {
		t.Fatalf("unstamped = %d, want exactly 1", len(unstamped))
	}
	o := unstamped[0]
	if o.SHA != "o1" {
		t.Errorf("orphan SHA = %q, want o1 (orphans must surface with their SHAs)", o.SHA)
	}
	if o.Subject != "chore: an unstamped landing" {
		t.Errorf("orphan Subject = %q, want the original subject preserved", o.Subject)
	}
	if o.Band != BandUnverifiable {
		t.Errorf("ungraded orphan Band = %q, want %q (debt must not read as cleared)", o.Band, BandUnverifiable)
	}
	if len(units) != 1 || len(units[0].Commits) != 1 {
		t.Fatalf("units = %+v, want exactly one single-commit unit alongside the orphan", units)
	}
}

// TestMembershipPropertyRandomCorpus is the property half of the coverage: over
// randomized corpora mixing stamped, unstamped, malformed, and body-stamped
// commits, the partition stays total and disjoint and the arithmetic invariant
// holds. Seeded, so a failure reproduces.
func TestMembershipPropertyRandomCorpus(t *testing.T) {
	rng := rand.New(rand.NewSource(5033))
	leaves := []string{"steerpr", "cache", "gateway", "relay", "model"}
	for trial := 0; trial < 20; trial++ {
		n := 1 + rng.Intn(40)
		var records []string
		wantOrphans := 0
		for i := 0; i < n; i++ {
			sha := fmt.Sprintf("t%d-c%d", trial, i)
			switch rng.Intn(4) {
			case 0: // well-formed stamp
				leaf := leaves[rng.Intn(len(leaves))]
				records = append(records, membershipRecord(sha,
					fmt.Sprintf("feat(%s): work item %d (fak %s)", leaf, i, leaf), ""))
			case 1: // no stamp
				records = append(records, membershipRecord(sha,
					fmt.Sprintf("chore: unstamped item %d", i), ""))
				wantOrphans++
			case 2: // malformed stamp (uppercase leaf never matches the grammar)
				records = append(records, membershipRecord(sha,
					fmt.Sprintf("fix(x): malformed item %d (fak BAD)", i), ""))
				wantOrphans++
			default: // stamp only in the body
				records = append(records, membershipRecord(sha,
					fmt.Sprintf("fix(x): body-stamped item %d", i), "(fak steerpr)"))
				wantOrphans++
			}
		}
		commits := ParseLog(strings.Join(records, ""))
		if len(commits) != n {
			t.Fatalf("trial %d: ParseLog() = %d commits, want %d", trial, len(commits), n)
		}
		// membershipPartition asserts totality + disjointness internally.
		_, _, _, _, unstamped := membershipPartition(t, commits)
		if len(unstamped) != wantOrphans {
			t.Errorf("trial %d: orphans = %d, want %d (every non-conforming stamp must be detected)",
				trial, len(unstamped), wantOrphans)
		}
	}
}
