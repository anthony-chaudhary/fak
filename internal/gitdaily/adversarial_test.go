package gitdaily

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestAdversarialGitExecutionErrorsStopTheTick(t *testing.T) {
	for _, tc := range []struct {
		name string
		code int
		err  error
		want string
	}{
		{name: "git refused", code: 128, want: "treedoctor"},
		{name: "git unavailable", err: errors.New("executable missing"), want: "treedoctor"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			calls := 0
			runner := func(context.Context, string, ...string) (string, int, error) {
				calls++
				return "hostile stderr", tc.code, tc.err
			}
			got := Run(context.Background(), runner, Options{
				RepoRoot: t.TempDir(), GitCommonDir: t.TempDir(), Apply: true, Force: true,
				IndexLockSweep: func(context.Context, string, time.Time, bool) IndexLockSweep { return IndexLockSweep{} },
			})
			if !got.Incident {
				t.Fatalf("Run result = %#v, want incident after runner failure (%s)", got, tc.want)
			}
			if calls == 0 {
				t.Fatal("runner failure fixture was not exercised")
			}
		})
	}
}

func TestAdversarialStatusBoundsAndScreensHostileLedger(t *testing.T) {
	path := filepath.Join(t.TempDir(), LedgerName)
	valid := `{"schema":"fak-git-daily/1","day":"2026-08-01","at":"2026-08-01T00:00:00Z"}`
	body := strings.Join([]string{
		"{not-json}",
		valid,
		`{"schema":"wrong","day":"2026-08-02"}`,
		`{"schema":"fak-git-daily/1","day":""}`,
		strings.Repeat("x", 70<<10),
	}, "\n") + "\n"
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	rows := Status(path, 1)
	if len(rows) != 1 || rows[0].Day != "2026-08-01" {
		t.Fatalf("Status(hostile ledger) = %#v, want only the one valid bounded row", rows)
	}
}

func TestEdgeLockSweepFailureIsAnIncident(t *testing.T) {
	got := Run(context.Background(), func(context.Context, string, ...string) (string, int, error) {
		return "hostile git refusal", 128, nil
	}, Options{
		RepoRoot: t.TempDir(), GitCommonDir: t.TempDir(), Apply: true, Force: true,
		IndexLockSweep: func(context.Context, string, time.Time, bool) IndexLockSweep {
			return IndexLockSweep{Err: "permission denied"}
		},
	})
	if !got.Incident || !strings.Contains(got.Locks.IndexErr, "permission denied") {
		t.Fatalf("lock sweep failure = %#v, want explicit incident", got)
	}
}
