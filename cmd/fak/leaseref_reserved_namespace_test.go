package main

import (
	"bytes"
	"encoding/hex"
	"fmt"
	"maps"
	"os/exec"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gardenbundle"
)

// Both user-invoked pruning and the automatic garden lease phase must preserve
// opaque coordination refs. Each path runs against its own disposable Git repo.
// fak-test:runtime integration est=1s lane=default
func TestReservedNamespaceCLIPruning(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	for _, path := range []string{"leaseref-reap", "garden-lease-phase"} {
		t.Run(path, func(t *testing.T) {
			t.Parallel()
			dir := t.TempDir()
			reservedNamespaceCLIGit(t, dir, "", "init", "-q")
			now := time.Now().UTC()
			encoded := hex.EncodeToString([]byte("fixture~ticket"))
			payloads := map[string]string{
				"contract/v1/" + encoded: fmt.Sprintf(`{"ticket_id":"fixture~ticket","holder":"fixture","ttl_seconds":3600,"acquired_at":%q,"renewed_at":%q}`,
					now.Add(-time.Minute).Format(time.RFC3339), now.Format(time.RFC3339)),
				"contract/v1/not-hex":          `{"acquired_unix":1,"ttl_seconds":1}`,
				"contract/future/v2/object":    `{"acquired_unix":1,"ttl_seconds":1}`,
				"contract/future/v2/malformed": `{"not":"complete"`,
				"epoch/v1/" + encoded:          `7`,
				"epoch/future/v2/object":       `{"acquired_unix":1,"ttl_seconds":1}`,
				"epoch/future/v2/hostile":      `{"id":"live-generic","acquired_unix":1,"ttl_seconds":1}`,
				"epoch/future/v2/malformed":    `not-json`,
				"contract-legacy":              `{"ticket_id":"legacy","state":"EXECUTING","acquired_unix":1,"ttl_seconds":1}`,
				"session-live":                 fmt.Sprintf(`{"id":"live","updated_at":%d,"ttl_seconds":3600}`, now.Unix()),
				"intent-issue-27":              fmt.Sprintf(`{"target":"#27","key":"issue-27","acquired_unix":%d,"ttl_seconds":3600}`, now.Unix()),
				"expired-generic":              `{"id":"expired-generic","acquired_unix":1,"ttl_seconds":1}`,
				"live-generic":                 fmt.Sprintf(`{"id":"live-generic","acquired_unix":%d,"ttl_seconds":3600}`, now.Unix()),
			}
			for suffix, payload := range payloads {
				oid := reservedNamespaceCLIGit(t, dir, payload, "hash-object", "-w", "--stdin")
				reservedNamespaceCLIGit(t, dir, "", "update-ref", "refs/fak/locks/"+suffix, oid)
			}
			want := reservedNamespaceCLIRefs(t, dir)
			delete(want, "refs/fak/locks/expired-generic")
			var out, stderr bytes.Buffer
			switch path {
			case "leaseref-reap":
				if code := runLeaseref(&out, &stderr, []string{"reap", "--dir", dir}); code != 0 {
					t.Errorf("leaseref reap exited %d: %s", code, stderr.String())
				}
				if !strings.Contains(out.String(), "reaped 1 expired lease(s), 0 expired session(s), 0 lapsed intent(s)") {
					t.Errorf("unexpected reap counts: %s", out.String())
				}
			case "garden-lease-phase":
				// Exercise the actual destructive garden phase without unrelated
				// growth/sentinel cleanup or dispatching work from a full tick.
				plan := gardenbundle.TickPlan{Decisions: []gardenbundle.ActDecision{
					{Act: gardenbundle.ActReap, Perform: true},
				}}
				var counts gardenTickCounts
				if err := gardenPhaseReapLeases(&stderr, plan, dir, &counts); err != nil {
					t.Errorf("garden lease phase: %v", err)
				}
				if counts.Reaped != 1 || counts.Sessions != 0 {
					t.Errorf("garden reap counts = %+v; want 1 generic lease and no sessions", counts)
				}
			}
			if stderr.Len() != 0 {
				t.Errorf("%s stderr: %s", path, stderr.String())
			}
			if got := reservedNamespaceCLIRefs(t, dir); !maps.Equal(got, want) {
				t.Errorf("%s changed protected ref OIDs: got %v, want %v", path, got, want)
			}
		})
	}
}

func reservedNamespaceCLIGit(t *testing.T, dir, stdin string, args ...string) string {
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

func reservedNamespaceCLIRefs(t *testing.T, dir string) map[string]string {
	t.Helper()
	out := reservedNamespaceCLIGit(t, dir, "", "for-each-ref", "--format=%(refname) %(objectname)", "refs/fak/locks/")
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
