package safesync

import (
	"context"
	"errors"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func localOnlyFixture(t *testing.T) (string, string, *ReconciliationPacket) {
	t.Helper()
	origin, clone := setupTestOriginAndClone(t)
	writeFile(t, filepath.Join(origin, "remote-local-only.txt"), "remote\n")
	git(t, origin, "add", ".")
	git(t, origin, "commit", "-m", "remote change")
	writeFile(t, filepath.Join(clone, "local-local-only.txt"), "local\n")
	// Real Go sources make the default validation witness execute go build/test,
	// rather than take the no-Go-source early return.
	writeFile(t, filepath.Join(clone, "go.mod"), "module example.com/localcheckpoint\n\ngo 1.22\n")
	writeFile(t, filepath.Join(clone, "checkpoint.go"), "package checkpoint\nfunc Value() int { return 7 }\n")
	writeFile(t, filepath.Join(clone, "checkpoint_test.go"), "package checkpoint\nimport \"testing\"\nfunc TestValue(t *testing.T) { if Value() != 7 { t.Fatal(Value()) } }\n")
	git(t, clone, "add", ".")
	git(t, clone, "commit", "-m", "local change")
	git(t, clone, "fetch", "origin")
	packet, err := BuildReconciliationPacket(context.Background(), PacketOptions{Repo: clone, Remote: "origin", Branch: "work"})
	if err != nil || packet == nil || packet.Disposition != DispositionSafeDisjoint {
		t.Fatalf("packet = %+v, err = %v", packet, err)
	}
	return origin, clone, packet
}

func TestExecutePacketLocalOnlyCheckpoint(t *testing.T) {
	t.Setenv("GOWORK", "off")
	t.Setenv("GOTOOLCHAIN", "local")
	origin, clone, packet := localOnlyFixture(t)
	builds, tests, transportCalls := 0, 0, 0
	// Omitting witnesses must not turn local-only into an unvalidated checkpoint.
	packet.RequiredWitnesses = nil
	runner := func(ctx context.Context, repo string, args ...string) RunResult {
		if len(args) > 0 && (args[0] == "push" || args[0] == "fetch") {
			transportCalls++
			return RunResult{Code: 1, Err: errors.New("unexpected executor Git transport")}
		}
		return RealRunner(ctx, repo, args...)
	}
	receipt, err := ExecutePacket(context.Background(), packet, ExecuteOptions{
		Repo: clone, Remote: "origin", Branch: "work", LocalOnly: true, Runner: runner,
		BuildVerifier: func(ctx context.Context, repo string) error { builds++; return defaultBuildVerifier(ctx, repo) },
		TestVerifier:  func(ctx context.Context, repo string) error { tests++; return defaultTestVerifier(ctx, repo) },
	})
	if err != nil || receipt == nil || receipt.Status != ExecuteStatusExecuted || receipt.Pushed || !receipt.LocalOnly || !receipt.LocalCommitsContained {
		t.Fatalf("receipt=%+v err=%v", receipt, err)
	}
	if builds != 1 || tests != 1 || transportCalls != 0 {
		t.Fatalf("builds=%d tests=%d transportCalls=%d", builds, tests, transportCalls)
	}
	remote, err := rev(context.Background(), RealRunner, origin, "HEAD")
	if err != nil || remote != packet.TargetSHA {
		t.Fatalf("remote changed: %s (%v)", remote, err)
	}
	parents := RealRunner(context.Background(), clone, "rev-list", "--parents", "-n", "1", receipt.NewHEAD)
	if got, want := strings.TrimSpace(string(parents.Stdout)), receipt.NewHEAD+" "+packet.LocalHead+" "+packet.TargetSHA; got != want {
		t.Fatalf("parents=%q want=%q", got, want)
	}
}

func TestExecutePacketLocalOnlyValidationFailure(t *testing.T) {
	for _, gate := range []string{"build", "test", "dirty-after-validation"} {
		t.Run(gate, func(t *testing.T) {
			origin, clone, packet := localOnlyFixture(t)
			opts := ExecuteOptions{Repo: clone, Branch: "work", LocalOnly: true,
				BuildVerifier: func(context.Context, string) error { return nil },
				TestVerifier:  func(context.Context, string) error { return nil },
			}
			want := ReasonBuildCheckFailed
			switch gate {
			case "build":
				opts.BuildVerifier = func(context.Context, string) error { return errors.New("build failed") }
			case "test":
				want = ReasonTestCheckFailed
				opts.TestVerifier = func(context.Context, string) error { return errors.New("test failed") }
			default:
				want = ReasonPathspecRace
				opts.TestVerifier = func(context.Context, string) error {
					writeFile(t, filepath.Join(clone, "unexpected.txt"), "do not sweep into checkpoint\n")
					return nil
				}
			}
			receipt, err := ExecutePacket(context.Background(), packet, opts)
			if err == nil || receipt == nil || receipt.Status != ExecuteStatusFailed || receipt.Pushed || receipt.Reason != want {
				t.Fatalf("receipt=%+v err=%v", receipt, err)
			}
			remote, err := rev(context.Background(), RealRunner, origin, "HEAD")
			if err != nil || remote != packet.TargetSHA {
				t.Fatalf("remote changed: %s (%v)", remote, err)
			}
		})
	}
}

func TestExecutePacketLocalOnlyPreservesRefusals(t *testing.T) {
	for _, kind := range []string{"owner-handoff", "dirty", "staged", "writer-held", "head-race"} {
		t.Run(kind, func(t *testing.T) {
			_, clone, packet := localOnlyFixture(t)
			opts := ExecuteOptions{Repo: clone, Branch: "work", LocalOnly: true}
			switch kind {
			case "owner-handoff":
				packet.Disposition, packet.Dispatchable = DispositionOwnerHandoffRequired, false
			case "writer-held":
				lease, err := AcquireWriterLease(clone, "peer-writer", time.Now, DefaultWriterLeaseTTL)
				if err != nil {
					t.Fatal(err)
				}
				defer lease.Release()
			case "dirty", "staged":
				writeFile(t, filepath.Join(clone, "dirty.txt"), "peer bytes\n")
				packet.DirtyPaths = []string{"dirty.txt"}
				if kind == "staged" {
					git(t, clone, "add", "dirty.txt")
				}
			case "head-race":
				heads := 0
				opts.Runner = func(ctx context.Context, repo string, args ...string) RunResult {
					if len(args) > 0 && args[0] == "rev-parse" && args[len(args)-1] == "HEAD" {
						heads++
						if heads == 2 {
							return RunResult{Stdout: []byte(packet.TargetSHA + "\n")}
						}
					}
					return RealRunner(ctx, repo, args...)
				}
			}
			receipt, err := ExecutePacket(context.Background(), packet, opts)
			if err == nil || receipt == nil || receipt.Status != ExecuteStatusRefused || receipt.Pushed {
				t.Fatalf("receipt=%+v err=%v", receipt, err)
			}
			head, _ := rev(context.Background(), RealRunner, clone, "HEAD")
			if head != packet.LocalHead {
				t.Fatalf("refused operation changed HEAD to %s", head)
			}
		})
	}
}

func TestReconcileLocalOnlyRejectsApply(t *testing.T) {
	for _, opts := range []ReconcileOptions{{LocalOnly: true}, {LocalOnly: true, Execute: true, Apply: true}} {
		opts.Runner = func(context.Context, string, ...string) RunResult {
			t.Fatal("invalid local-only options reached git")
			return RunResult{}
		}
		if _, err := RouteReconciliation(context.Background(), opts); err == nil {
			t.Fatal("expected option refusal")
		}
	}
}

func TestExecutePacketLocalOnlyReadbackBoundary(t *testing.T) {
	for _, failure := range []string{"head-drift", "lease-loss"} {
		t.Run(failure, func(t *testing.T) {
			_, clone, packet := localOnlyFixture(t)
			git(t, clone, "merge", "--no-ff", "--no-edit", packet.TargetSHA)
			newHEAD, err := rev(context.Background(), RealRunner, clone, "HEAD")
			if err != nil {
				t.Fatal(err)
			}
			lease, err := AcquireWriterLease(clone, "local-only-readback-test", time.Now, DefaultWriterLeaseTTL)
			if err != nil {
				t.Fatal(err)
			}
			defer lease.Release()
			injected := false
			runner := func(ctx context.Context, repo string, args ...string) RunResult {
				res := RealRunner(ctx, repo, args...)
				if len(args) > 0 && args[0] == "rev-list" && !injected {
					injected = true
					// Drift happens AFTER the successful parent query. A pre-query
					// check alone would incorrectly return EXECUTED.
					if failure == "head-drift" {
						git(t, clone, "update-ref", "HEAD", packet.LocalHead, newHEAD)
					} else {
						lease.markLost()
					}
				}
				return res
			}
			e := NewPacketExecutor(ExecuteOptions{Repo: clone, Branch: "work", LocalOnly: true, Runner: runner})
			receipt, err := e.localOnlyReadback(context.Background(), packet, packet.TargetRef, newHEAD, lease)
			want := ReasonPathspecRace
			if failure == "lease-loss" {
				want = ReasonLeaseOwnerUnavailable
			}
			if !injected || err == nil || receipt == nil || receipt.Status != ExecuteStatusFailed || receipt.NewHEAD != newHEAD || receipt.Reason != want || receipt.Pushed {
				t.Fatalf("injected=%v receipt=%+v err=%v", injected, receipt, err)
			}
		})
	}
}
