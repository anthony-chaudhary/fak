package safecommit

import (
	"context"
	"reflect"
	"sort"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/wipref"
	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

type incidentalMessage func(t *testing.T, base string) string

func incidentalScoped(scope ...string) incidentalMessage {
	return func(t *testing.T, base string) string { return peerHistoryStamp(t, "peer-a", base, scope) }
}

// incidentalCaptureFixture models a shared working tree: one peer checkpoint that
// declares a narrow scope while its tree-wide capture also holds another session's
// dirty and untracked files.
func incidentalCaptureFixture(t *testing.T, message incidentalMessage) *peerHistoryFixture {
	t.Helper()
	t.Setenv(workerworktree.WorktreeDirEnv, "")
	f := newPeerHistoryFixture(t)
	base := f.commit(nil, "base", map[string]peerHistoryEntry{
		"peer/owned.go": {mode: "100644", body: "base\n"},
		"mine/a.go":     {mode: "100644", body: "base\n"},
	})
	f.git("update-ref", "HEAD", base)
	f.checkpoint(t, "peer-a", message(t, base), base)
	return f
}

// checkpoint anchors a tree-wide capture of the shared dirty tree under ref and
// leaves the working tree holding that content with the index back at HEAD.
func (f *peerHistoryFixture) checkpoint(t *testing.T, ref, message, base string) {
	t.Helper()
	oid := f.commit([]string{base}, message, map[string]peerHistoryEntry{
		"peer/owned.go": {mode: "100644", body: "peer edit\n"},
		"mine/a.go":     {mode: "100644", body: "my edit\n"},
		"mine/new.go":   {mode: "100644", body: "my new file\n"},
	})
	f.git("update-ref", "refs/fak/wip/"+ref, oid)
	f.git("read-tree", base)
}

func validateIncidental(t *testing.T, f *peerHistoryFixture, opts PathAttributionOptions, paths ...string) PathAttributionResult {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	opts.SessionID = "self"
	res, err := ValidatePathAttribution(ctx, realRunner, f.dir, paths, opts)
	if err != nil {
		t.Fatalf("ValidatePathAttribution(%v): %v", paths, err)
	}
	return res
}

func requireIncidentalAdmit(t *testing.T, res PathAttributionResult, want ...string) {
	t.Helper()
	if !res.OK || len(res.CollidingPaths) != 0 {
		t.Fatalf("a peer capture outside its declared scope refused the author's own exact paths: %+v", res)
	}
	var got []string
	for _, rec := range res.ReconciledPaths {
		if rec.Classification != OwnershipIncidentalCapture || rec.PeerSession != "peer-a" || rec.CheckpointOID == "" {
			t.Fatalf("incidental capture not recorded as evidence: %+v", rec)
		}
		got = append(got, rec.Path)
	}
	sort.Strings(got)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("reconciled paths = %v, want %v", got, want)
	}
}

func requireCollision(t *testing.T, res PathAttributionResult, peers ...string) {
	t.Helper()
	if res.OK || res.Reason != ReasonPeerWIPCollision || !reflect.DeepEqual(res.PeerSessions, peers) {
		t.Fatalf("want %s owned by %v, got %+v", ReasonPeerWIPCollision, peers, res)
	}
}

// fak-test:runtime medium est=30s lane=default
func TestPeerWIPDeclaredScopeDoesNotClaimIncidentalCapture(t *testing.T) {
	scoped := incidentalScoped("peer/owned.go")

	t.Run("ExactPathsOutsideDeclaredScopeAreAdmitted", func(t *testing.T) {
		f := incidentalCaptureFixture(t, scoped)
		owners, err := f.resolve(t, realRunner, []string{"mine/a.go", "mine/new.go", "peer/owned.go"})
		if err != nil {
			t.Fatal(err)
		}
		if !owners["mine/a.go"].Incidental || !owners["mine/new.go"].Incidental || owners["peer/owned.go"].Incidental {
			t.Fatalf("incidental flags wrong: %+v", owners)
		}
		requireIncidentalAdmit(t, validateIncidental(t, f, PathAttributionOptions{}, "mine/a.go", "mine/new.go"), "mine/a.go", "mine/new.go")
	})

	t.Run("DirectoryDeclaredScope", func(t *testing.T) {
		f := incidentalCaptureFixture(t, incidentalScoped("peer"))
		requireIncidentalAdmit(t, validateIncidental(t, f, PathAttributionOptions{}, "mine/a.go"), "mine/a.go")
		requireCollision(t, validateIncidental(t, f, PathAttributionOptions{}, "peer/owned.go"), "peer-a")
	})

	t.Run("ManagedWorkerIsAdmittedWithoutHunkReconcile", func(t *testing.T) {
		f := incidentalCaptureFixture(t, scoped)
		requireIncidentalAdmit(t, validateIncidental(t, f, PathAttributionOptions{ManagedWorker: true}, "mine/a.go"), "mine/a.go")
	})

	t.Run("WitnessedConflictingHunkStillRefuses", func(t *testing.T) {
		f := incidentalCaptureFixture(t, scoped)
		res := validateIncidental(t, f, PathAttributionOptions{
			ManagedWorker:   true,
			WitnessedDeltas: []WitnessedPathDelta{{Path: "mine/a.go", Status: OwnershipConflictingHunk}},
		}, "mine/a.go")
		requireCollision(t, res, "peer-a")
	})

	t.Run("DeclaredScopeStillCollides", func(t *testing.T) {
		f := incidentalCaptureFixture(t, scoped)
		res := validateIncidental(t, f, PathAttributionOptions{}, "peer/owned.go")
		requireCollision(t, res, "peer-a")
		if !reflect.DeepEqual(res.CollidingPaths, []string{"peer/owned.go"}) {
			t.Fatalf("colliding paths = %v", res.CollidingPaths)
		}
	})

	t.Run("DirectorySweepStillCollides", func(t *testing.T) {
		f := incidentalCaptureFixture(t, scoped)
		requireCollision(t, validateIncidental(t, f, PathAttributionOptions{}, "mine"), "peer-a")
		requireCollision(t, validateIncidental(t, f, PathAttributionOptions{}, "mine/a.go", "mine"), "peer-a")
	})

	t.Run("UndeclaredCaptureKeepsItsClaim", func(t *testing.T) {
		stamp := func(mutate func(*wipref.Stamp)) incidentalMessage {
			return func(t *testing.T, base string) string {
				s := wipref.Stamp{SessionID: "peer-a", StartSHA: base, Scope: []string{"peer/owned.go"}}
				mutate(&s)
				msg, err := wipref.EncodeStamp(s)
				if err != nil {
					t.Fatal(err)
				}
				return msg
			}
		}
		for name, message := range map[string]incidentalMessage{
			"NoStamp":              func(*testing.T, string) string { return "legacy checkpoint" },
			"EmptyScope":           stamp(func(s *wipref.Stamp) { s.Scope = nil }),
			"ForeignSession":       stamp(func(s *wipref.Stamp) { s.SessionID = "someone-else" }),
			"MissingStartSHA":      stamp(func(s *wipref.Stamp) { s.StartSHA = "" }),
			"MetadataOnly":         stamp(func(s *wipref.Stamp) { s.MetadataOnly = true }),
			"NonCanonicalScope":    stamp(func(s *wipref.Stamp) { s.Scope = []string{"./peer/owned.go"} }),
			"ScopeMatchesNoChange": stamp(func(s *wipref.Stamp) { s.Scope = []string{"peer/typo.go"} }),
			"StampWithProse": func(t *testing.T, base string) string {
				return peerHistoryStamp(t, "peer-a", base, []string{"peer/owned.go"}) + "\nfak-wip: {}"
			},
		} {
			t.Run(name, func(t *testing.T) {
				f := incidentalCaptureFixture(t, message)
				requireCollision(t, validateIncidental(t, f, PathAttributionOptions{}, "mine/a.go"), "peer-a")
			})
		}
	})

	t.Run("RealClaimOutranksIncidentalCapture", func(t *testing.T) {
		for ref, message := range map[string]incidentalMessage{
			// Sorts before peer-a: an earlier real claim is never displaced.
			"a-legacy": func(*testing.T, string) string { return "legacy checkpoint" },
			// Sort after peer-a: a later real claim replaces the incidental slot.
			"z-legacy": func(*testing.T, string) string { return "legacy checkpoint" },
			"z-scoped": func(t *testing.T, base string) string {
				return peerHistoryStamp(t, "z-scoped", base, []string{"mine/a.go"})
			},
		} {
			t.Run(ref, func(t *testing.T) {
				f := incidentalCaptureFixture(t, scoped)
				base := f.git("rev-parse", "HEAD")
				f.checkpoint(t, ref, message(t, base), base)

				owners, err := f.resolve(t, realRunner, []string{"mine/a.go", "peer/owned.go"})
				if err != nil {
					t.Fatal(err)
				}
				if got := owners["mine/a.go"]; got.Peer != ref || got.Incidental {
					t.Fatalf("mine/a.go owner = %+v, want the real %s claim", got, ref)
				}
				requireCollision(t, validateIncidental(t, f, PathAttributionOptions{}, "mine/a.go"), ref)
			})
		}
	})
}
