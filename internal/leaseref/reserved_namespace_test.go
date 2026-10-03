package leaseref

import (
	"context"
	"encoding/hex"
	"fmt"
	"maps"
	"os/exec"
	"slices"
	"strings"
	"testing"
	"time"
)

// Reserved subtrees are opaque, even when their payloads resemble generic leases.
// Real Git matters: the production batch deleter accepts nested ref names that
// the non-stdin Release path rejects, so a Runner-only test misses ref deletion.
// fak-test:runtime integration est=2s lane=default
func TestReservedNamespaceLeaseReadersAndReap(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	for _, mode := range []string{"NewInDir", "batch-read-fallback", "without-stdin"} {
		t.Run(mode, func(t *testing.T) {
			t.Parallel()
			dir := t.TempDir()
			reservedNamespaceGit(t, dir, "", "init", "-q")
			now := time.Unix(1800000000, 0).UTC()
			encoded := hex.EncodeToString([]byte("fixture~ticket"))
			payloads := map[string]string{
				"contract/v1/" + encoded: fmt.Sprintf(`{"ticket_id":"fixture~ticket","holder":"fixture","ttl_seconds":3600,"acquired_at":%q,"renewed_at":%q}`,
					now.Add(-time.Minute).Format(time.RFC3339), now.Format(time.RFC3339)),
				"contract/v1/not-hex":          `{"acquired_unix":1,"ttl_seconds":1}`,
				"contract/future/v2/object":    `{"future_schema":99,"ttl_seconds":0}`,
				"contract/future/v2/malformed": `{"not":"complete"`,
				"epoch/v1/" + encoded:          `7`,
				"epoch/future/v2/object":       `{"acquired_unix":1,"ttl_seconds":1}`,
				// A payload-supplied ID must not redirect a reap to a live generic ref.
				"epoch/future/v2/hostile":   `{"id":"live-generic","acquired_unix":1,"ttl_seconds":1}`,
				"epoch/future/v2/malformed": `not-json`,
				"contract-legacy":           `{"ticket_id":"legacy","state":"EXECUTING","acquired_unix":1,"ttl_seconds":1}`,
				"session-legacy":            `{"id":"legacy","updated_at":1,"ttl_seconds":1}`,
				"intent-issue-27":           `{"target":"#27","key":"issue-27","acquired_unix":1,"ttl_seconds":1}`,
				"expired-generic":           `{"id":"expired-generic","acquired_unix":1,"ttl_seconds":1}`,
				"live-generic":              fmt.Sprintf(`{"id":"live-generic","acquired_unix":%d,"ttl_seconds":3600}`, now.Unix()),
				// Similar names outside the reserved subtrees remain generic leases.
				"contracts-backup": `{"id":"contracts-backup","acquired_unix":1,"ttl_seconds":1}`,
				"epoch-backup":     `{"id":"epoch-backup","acquired_unix":1,"ttl_seconds":1}`,
			}
			for suffix, payload := range payloads {
				oid := reservedNamespaceGit(t, dir, payload, "hash-object", "-w", "--stdin")
				reservedNamespaceGit(t, dir, "", "update-ref", "refs/fak/locks/"+suffix, oid)
			}
			before := reservedNamespaceRefs(t, dir)
			assertRefs := func(stage string, want map[string]string) {
				t.Helper()
				if got := reservedNamespaceRefs(t, dir); !maps.Equal(got, want) {
					t.Errorf("%s changed protected ref OIDs: got %v, want %v", stage, got, want)
				}
			}
			assertOpaque := func(ref string) {
				t.Helper()
				if strings.HasPrefix(ref, "refs/fak/locks/contract/") || strings.HasPrefix(ref, "refs/fak/locks/epoch/") {
					t.Errorf("reserved ref reached payload reader: %s", ref)
				}
			}
			var batchReads, plainReads, batchDeletes, plainDeletes int
			run := func(ctx context.Context, dir string, args ...string) (string, int, error) {
				if len(args) >= 3 && args[0] == "cat-file" {
					plainReads++
					assertOpaque(args[len(args)-1])
				}
				if len(args) >= 2 && args[0] == "update-ref" && args[1] == "-d" {
					plainDeletes++
				}
				return gitRunner(ctx, dir, args...)
			}
			stdinRun := func(ctx context.Context, dir, stdin string, args ...string) (string, int, error) {
				if len(args) >= 2 && args[0] == "cat-file" && args[1] == "--batch" {
					batchReads++
					for _, ref := range strings.Fields(stdin) {
						assertOpaque(ref)
					}
					if mode == "batch-read-fallback" {
						return "", 1, nil
					}
				}
				if len(args) >= 2 && args[0] == "update-ref" && args[1] == "--stdin" {
					batchDeletes++
				}
				return gitStdinRunner(ctx, dir, stdin, args...)
			}
			var store *Store
			switch mode {
			case "NewInDir":
				store = NewInDir(dir)
				// Observe the real runners without replacing their Git operations.
				store.run, store.runStdin = run, stdinRun
			case "batch-read-fallback":
				store = NewWithStdinRunner(run, stdinRun, dir)
			case "without-stdin":
				store = NewWithRunner(run, dir)
			}
			ctx := context.Background()
			wantExpired := []string{"contracts-backup", "epoch-backup", "expired-generic"}
			wantAll := append(slices.Clone(wantExpired), "live-generic")
			all, err := store.List(ctx)
			if err != nil || !slices.Equal(reservedNamespaceIDs(all), wantAll) {
				t.Errorf("List = %v, %v; want %v", reservedNamespaceIDs(all), err, wantAll)
			}
			assertRefs("List", before)
			strict, err := store.StrictSnapshot(ctx)
			if err != nil || !slices.Equal(reservedNamespaceIDs(strict), wantAll) {
				t.Errorf("StrictSnapshot = %v, %v; want %v", reservedNamespaceIDs(strict), err, wantAll)
			}
			assertRefs("StrictSnapshot", before)
			live, expired, err := store.Live(ctx, now)
			if err != nil || !slices.Equal(reservedNamespaceIDs(live), []string{"live-generic"}) || !slices.Equal(expired, wantExpired) {
				t.Errorf("Live = %v, expired = %v, err = %v; want [live-generic], %v", reservedNamespaceIDs(live), expired, err, wantExpired)
			}
			assertRefs("Live", before)
			reaped, err := store.Reap(ctx, now)
			if err != nil || !slices.Equal(reaped, wantExpired) {
				t.Errorf("Reap = %v, %v; want %v", reaped, err, wantExpired)
			}
			wantAfter := maps.Clone(before)
			for _, id := range wantExpired {
				delete(wantAfter, "refs/fak/locks/"+id)
			}
			assertRefs("Reap", wantAfter)
			if mode == "without-stdin" {
				if batchReads != 0 || plainReads == 0 || batchDeletes != 0 || plainDeletes != len(wantExpired) {
					t.Errorf("non-stdin path: reads batch/plain=%d/%d, deletes batch/plain=%d/%d", batchReads, plainReads, batchDeletes, plainDeletes)
				}
			} else if batchReads == 0 || batchDeletes != 1 || plainDeletes != 0 || (mode == "batch-read-fallback" && plainReads == 0) {
				t.Errorf("%s path: reads batch/plain=%d/%d, deletes batch/plain=%d/%d", mode, batchReads, plainReads, batchDeletes, plainDeletes)
			}
			if again, err := store.Reap(ctx, now); err != nil || len(again) != 0 {
				t.Errorf("second Reap = %v, %v; want no changes", again, err)
			}
			assertRefs("second Reap", wantAfter)

			// Opaque subtrees must not become legacy contract/session/intent records.
			contracts, err := store.ListContracts(ctx)
			if err != nil || len(contracts) != 1 || contracts[0].TicketID != "legacy" {
				t.Errorf("ListContracts = %+v, %v; want only legacy", contracts, err)
			}
			sessions, err := store.ListSessions(ctx)
			if err != nil || len(sessions) != 1 || sessions[0].ID != "legacy" {
				t.Errorf("ListSessions = %+v, %v; want only legacy", sessions, err)
			}
			intents, err := store.ListIntents(ctx)
			if err != nil || len(intents) != 1 || intents[0].Key != "issue-27" {
				t.Errorf("ListIntents = %+v, %v; want only issue-27", intents, err)
			}
			assertRefs("typed readers", wantAfter)
		})
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestReservedNamespaceRefBoundaries(t *testing.T) {
	for _, tc := range []struct {
		ref  string
		want bool
	}{
		{"refs/fak/locks/contract", true},
		{"refs/fak/locks/epoch", true},
		{"refs/fak/locks/contractor/future", true},
		{"refs/fak/locks/epochs", true},
		{"refs/fak/locks/contract/v1/00", false},
		{"refs/fak/locks/contract/v1/not-hex", false},
		{"refs/fak/locks/contract/unknown/deeper", false},
		{"refs/fak/locks/epoch/v1/00", false},
		{"refs/fak/locks/epoch/unknown/deeper", false},
		{"refs/fak/locks/contract-legacy", false},
		{"refs/fak/locks/session-legacy", false},
		{"refs/fak/locks/intent-legacy", false},
		{"refs/other/locks/contract/v1/00", false},
	} {
		if got := isLeaseRef(tc.ref); got != tc.want {
			t.Errorf("isLeaseRef(%q) = %v, want %v", tc.ref, got, tc.want)
		}
	}
}

func reservedNamespaceIDs(records []Record) []string {
	ids := make([]string, 0, len(records))
	for _, record := range records {
		ids = append(ids, record.ID)
	}
	return ids
}

func reservedNamespaceGit(t *testing.T, dir, stdin string, args ...string) string {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	cmd.Stdin = strings.NewReader(stdin)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("fixture git %v: %v\n%s", args, err, out)
	}
	return strings.TrimSpace(string(out))
}

func reservedNamespaceRefs(t *testing.T, dir string) map[string]string {
	t.Helper()
	out := reservedNamespaceGit(t, dir, "", "for-each-ref", "--format=%(refname) %(objectname)", "refs/fak/locks/")
	refs := make(map[string]string)
	for _, line := range strings.Split(out, "\n") {
		if line == "" {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) != 2 {
			t.Fatalf("unexpected fixture ref row %q", line)
		}
		refs[fields[0]] = fields[1]
	}
	return refs
}
