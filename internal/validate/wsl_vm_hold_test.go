package validate

import (
	"context"
	"errors"
	"slices"
	"strings"
	"testing"
	"time"
)

func stubValidateWSLHoldClock(t *testing.T) time.Time {
	t.Helper()
	base := time.Now()
	prev := validateNow
	validateNow = func() time.Time { return base }
	t.Cleanup(func() { validateNow = prev })
	return base
}

// fak-test:runtime fast est=100ms lane=default
func TestValidateWSLVMHoldOutlivesRunContextUntilRelease(t *testing.T) {
	base := stubValidateWSLHoldClock(t)
	started := make(chan []string, 1)
	stopped := make(chan struct{})
	prev := validateWSLCommand
	t.Cleanup(func() { validateWSLCommand = prev })
	validateWSLCommand = func(ctx context.Context, args ...string) ([]byte, error) {
		started <- args
		<-ctx.Done()
		close(stopped)
		return nil, ctx.Err()
	}
	runCtx, cancelRun := context.WithDeadline(context.Background(), base.Add(4*time.Minute))
	release := holdValidateWSLVM(runCtx)

	args := <-started
	if want := []string{"-e", "sleep", "300"}; !slices.Equal(args, want) {
		t.Fatalf("hold args = %q, want %q (run bound plus margin)", args, want)
	}
	cancelRun()
	select {
	case <-stopped:
		t.Fatal("hold ended with the run context; it must last through the deferred workspace cleanup")
	case <-time.After(50 * time.Millisecond):
	}
	release()
	select {
	case <-stopped:
	default:
		t.Fatal("release returned before the hold session ended")
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestValidateWSLWorkspaceLossTypesVMRestart(t *testing.T) {
	cause := errors.New("bash: line 1: cd: /tmp/fak-validate-1-2: No such file or directory")
	cases := []struct {
		name        string
		anchor      string
		probe       string
		probeErr    error
		wantProbes  int
		wantRestart bool
	}{
		{name: "new boot", anchor: "boot-a", probe: "boot-b\n", wantProbes: 1, wantRestart: true},
		{name: "same boot", anchor: "boot-a", probe: "boot-a\n", wantProbes: 1},
		{name: "probe failed", anchor: "boot-a", probeErr: errors.New("exit status 1"), wantProbes: 1},
		{name: "unknown anchor", anchor: "", wantProbes: 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			prev := validateWSLCommand
			t.Cleanup(func() { validateWSLCommand = prev })
			probes := 0
			validateWSLCommand = func(_ context.Context, args ...string) ([]byte, error) {
				probes++
				if want := []string{"-e", "cat", validateWSLBootIDPath}; !slices.Equal(args, want) {
					t.Errorf("probe args = %q, want %q", args, want)
				}
				return []byte(tc.probe), tc.probeErr
			}
			err := typeValidateWSLWorkspaceLoss(context.Background(), tc.anchor, "/tmp/fak-validate-1-2", cause)
			if probes != tc.wantProbes {
				t.Fatalf("probes = %d, want %d", probes, tc.wantProbes)
			}
			if !errors.Is(err, cause) {
				t.Fatalf("err = %v, want the original cause preserved", err)
			}
			if got := errors.Is(err, errValidateWSLVMRestarted); got != tc.wantRestart {
				t.Fatalf("errors.Is(err, errValidateWSLVMRestarted) = %v, want %v", got, tc.wantRestart)
			}
		})
	}
}

// fak-test:runtime fast est=500ms lane=default
func TestExtractCommittedTipWSLReportsWorkspaceBoot(t *testing.T) {
	repo := t.TempDir()
	fake := installWSLRetryTestFake(t, func(int) ([]byte, error) {
		return []byte(validateWSLBootIDPrefix + "boot-a\n"), nil
	})
	ctx, cancel := context.WithTimeout(context.Background(), wslRetryTestTimeout)
	defer cancel()

	dir, bootID, err := extractCommittedTipWSLWithin(ctx, repo, wslRetryTestTip)
	if err != nil || dir == "" {
		t.Fatalf("extract dir=%q err=%v", dir, err)
	}
	if bootID != "boot-a" {
		t.Fatalf("boot id = %q, want boot-a", bootID)
	}
	scripts := wslRetryTestScripts(t, fake, repo)
	if len(scripts) != 1 || !strings.Contains(scripts[0], validateWSLBootIDPath) {
		t.Fatalf("materialize script does not read the boot id: %q", scripts)
	}
}
