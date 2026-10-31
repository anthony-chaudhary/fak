package workerworktree

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

// TestLandSourceFenceAtomicCommit uses real Git repositories and ref transactions.
// The race hook runs AFTER the source snapshot, not merely before preflight: an
// implementation that checks the generation and then uses plain trunk CAS fails.
func TestLandSourceFenceAtomicCommit(t *testing.T) {
	t.Setenv(IsolatedLandEnv, "1")
	ctx := context.Background()
	setup := func(t *testing.T) (preparedFixture, *leaseref.Store, SourceLeaseFence) {
		f := newPreparedFixture(t, "feature.txt")
		s := leaseref.NewInDir(f.root)
		r, v, err := s.AcquireFenced(ctx, leaseref.Record{ID: "source-test", Holder: "owner-a", TreeGlobs: []string{f.rel}, TTLSeconds: 3600}, time.Now())
		if err != nil || !v.OK || r.Generation <= 0 {
			t.Fatalf("source acquire: %+v %+v %v", r, v, err)
		}
		return f, s, SourceLeaseFence{ID: r.ID, Holder: r.Holder, Generation: r.Generation}
	}
	wantUnchanged := func(t *testing.T, f preparedFixture) {
		t.Helper()
		if head := strings.TrimSpace(mustGit(t, f.root, "rev-parse", "main")); head != f.base {
			t.Fatalf("refused attempt changed trunk: %s != %s", head, f.base)
		}
		if _, err := os.Stat(filepath.Join(f.root, f.rel)); !os.IsNotExist(err) {
			t.Fatalf("refused attempt wrote shared checkout: %v", err)
		}
	}
	release := func(t *testing.T, s *leaseref.Store, token SourceLeaseFence) {
		t.Helper()
		v, err := s.ReleaseFenced(ctx, token.ID, token.Holder, token.Generation, time.Now())
		if err != nil || !v.OK {
			t.Fatalf("release: %+v %v", v, err)
		}
	}
	supersede := func(t *testing.T, s *leaseref.Store, token SourceLeaseFence) {
		t.Helper()
		release(t, s, token)
		r, v, err := s.AcquireFenced(ctx, leaseref.Record{ID: token.ID, Holder: "owner-b", TreeGlobs: []string{"feature.txt"}, TTLSeconds: 3600}, time.Now())
		if err != nil || !v.OK || r.Generation <= token.Generation {
			t.Fatalf("supersede: %+v %+v %v", r, v, err)
		}
	}
	land := func(f preparedFixture, token SourceLeaseFence, opts ...LandOption) Result {
		opts = append([]LandOption{WithSourceLeaseFence(token)}, opts...)
		return Land(f.root, f.wt, f.base, f.msg, []string{f.rel}, nil, nil, opts...)
	}
	t.Run("active_token", func(t *testing.T) {
		f, _, token := setup(t)
		r := land(f, token)
		if !r.OK || !r.Committed || r.CommitSHA == "" {
			t.Fatalf("active source refused: %+v", r)
		}
		if got := strings.TrimSpace(mustGit(t, f.root, "rev-parse", "main")); got != r.CommitSHA {
			t.Fatalf("commit not on main: %s %+v", got, r)
		}
	})
	t.Run("prepared_lock_blocks_source_release", func(t *testing.T) {
		f, store, token := setup(t)
		checks := 0
		underLock := func(c *landConfig) {
			c.sourceCAS = func(root, input string, beforeCommit func(context.Context) error) (int, string) {
				return sourceFenceGitTransaction(root, input, func(ctx context.Context) error {
					checks++
					v, err := store.ReleaseFenced(ctx, token.ID, token.Holder, token.Generation, time.Now())
					if err != nil || v.OK || v.Reason != leaseref.ReasonLeaseContended {
						t.Errorf("source lock not held at commit decision: %+v %v", v, err)
					}
					return beforeCommit(ctx)
				})
			}
		}
		r := land(f, token, underLock)
		if !r.OK || !r.Committed || checks != 1 {
			t.Fatalf("prepared-lock active land: %+v checks=%d", r, checks)
		}
	})
	t.Run("stale_preflight", func(t *testing.T) {
		f, s, token := setup(t)
		supersede(t, s, token)
		r := land(f, token)
		if r.OK || r.Code != LandResultSourceFence {
			t.Fatalf("stale source admitted: %+v", r)
		}
		wantUnchanged(t, f)
	})
	for _, mutation := range []string{"supersede", "release", "renew", "symbolic_alias"} {
		t.Run("late_"+mutation, func(t *testing.T) {
			f, s, token := setup(t)
			calls := 0
			race := func(c *landConfig) {
				c.sourceCAS = func(root, input string, beforeCommit func(context.Context) error) (int, string) {
					calls++
					if !strings.Contains(input, "verify refs/fak/locks/"+token.ID+" ") || !strings.Contains(input, "update refs/heads/main ") {
						t.Fatalf("not one source+trunk transaction: %q", input)
					}
					switch mutation {
					case "supersede":
						supersede(t, s, token)
					case "release":
						release(t, s, token)
					case "symbolic_alias":
						oid := strings.TrimSpace(mustGit(t, root, "rev-parse", "refs/fak/locks/"+token.ID))
						mustGit(t, root, "update-ref", "refs/fak/locks/alias-target", oid)
						mustGit(t, root, "symbolic-ref", "refs/fak/locks/"+token.ID, "refs/fak/locks/alias-target")
					case "renew":
						_, v, err := s.RenewFenced(ctx, token.ID, token.Holder, token.Generation, 3601, time.Now())
						if err != nil || !v.OK {
							t.Fatalf("renew: %+v %v", v, err)
						}
					}
					return sourceFenceGitTransaction(root, input, beforeCommit)
				}
			}
			r := land(f, token, race)
			if r.OK || r.Code != LandResultSourceFenceContended || calls != 1 {
				t.Fatalf("late %s failed to fence: result=%+v transactions=%d", mutation, r, calls)
			}
			wantUnchanged(t, f)
			if mutation == "renew" {
				// A heartbeat is not a new owner. Re-reading its new OID under the
				// unchanged immutable tuple is safe on an explicitly retried call.
				if retry := land(f, token); !retry.OK || !retry.Committed {
					t.Fatalf("renewed same owner cannot land: %+v", retry)
				}
			}
		})
	}
	prepare := func(t *testing.T, f preparedFixture, token SourceLeaseFence) PreparedLandReceipt {
		t.Helper()
		r, result := PrepareProspectiveLand(f.root, f.wt, f.base, f.msg, []string{f.rel}, f.binding, nil,
			func(_ string, err error) Result { return Result{OK: err == nil, Detail: errorDetail(err)} }, nil, WithSourceLeaseFence(token))
		if !result.OK || r.SourceLease == nil || *r.SourceLease != token {
			t.Fatalf("prepare did not bind source: %+v %+v", result, r)
		}
		return r
	}
	t.Run("prepared_binding_and_active_accept", func(t *testing.T) {
		f, _, token := setup(t)
		r := prepare(t, f, token)
		want := preparedWant(f, r)
		bad := token
		bad.Generation++
		for _, opts := range [][]LandOption{nil, {WithSourceLeaseFence(bad)}} {
			got := AcceptPreparedLand(f.root, f.wt, want, nil, opts...)
			if got.OK || got.Code != LandResultSourceFence {
				t.Fatalf("prepared tuple downgraded: %+v", got)
			}
			wantUnchanged(t, f)
		}
		if got := AcceptPreparedLand(f.root, f.wt, want, nil, WithSourceLeaseFence(token)); !got.OK || !got.Committed {
			t.Fatalf("active prepared accept: %+v", got)
		}
	})
	t.Run("prepared_late_supersession", func(t *testing.T) {
		f, s, token := setup(t)
		r := prepare(t, f, token)
		calls := 0
		race := func(c *landConfig) {
			c.sourceCAS = func(root, input string, beforeCommit func(context.Context) error) (int, string) {
				calls++
				supersede(t, s, token)
				return sourceFenceGitTransaction(root, input, beforeCommit)
			}
		}
		got := AcceptPreparedLand(f.root, f.wt, preparedWant(f, r), nil, WithSourceLeaseFence(token), race)
		if got.OK || got.Code != LandResultSourceFenceContended || calls != 1 {
			t.Fatalf("prepared late CAS: %+v calls=%d", got, calls)
		}
		wantUnchanged(t, f)
	})
	t.Run("missing_lease_and_legacy_fallback_refused", func(t *testing.T) {
		f, s, token := setup(t)
		release(t, s, token)
		if got := land(f, token); got.OK || got.Code != LandResultSourceFence {
			t.Fatalf("missing source: %+v", got)
		}
		wantUnchanged(t, f)
		f, _, token = setup(t)
		t.Setenv(IsolatedLandEnv, "0")
		if got := land(f, token); got.OK || got.Code != LandResultSourceFence {
			t.Fatalf("unfenced fallback: %+v", got)
		}
		wantUnchanged(t, f)
	})
}
