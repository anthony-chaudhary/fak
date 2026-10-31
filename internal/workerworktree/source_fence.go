package workerworktree

import (
	"bufio"
	"bytes"
	"context"
	"encoding/hex"
	"fmt"
	"io"
	"os/exec"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const LandResultSourceFence = "LAND_SOURCE_FENCE_REFUSED"
const LandResultSourceFenceContended = "LAND_SOURCE_FENCE_CONTENDED"

// SourceLeaseFence is the immutable identity granted to one source owner. A
// heartbeat may change the ref's OID, but MUST NOT change this presented tuple.
// It is meaningful only in the same Git common store as the target branch.
type SourceLeaseFence struct {
	ID         string `json:"id"`
	Holder     string `json:"holder"`
	Generation int64  `json:"generation"`
}

// WithSourceLeaseFence opts this invocation into atomic same-repository source
// fencing. Existing unfenced binaries/attempts are not retroactively protected.
func WithSourceLeaseFence(f SourceLeaseFence) LandOption {
	return func(c *landConfig) { copy := f; c.sourceLease = &copy }
}

func sameSourceLease(a, b *SourceLeaseFence) bool {
	if a == nil || b == nil {
		return a == b
	}
	return *a == *b
}

func sourceFenceRefusal(reason, detail string) Result {
	return Result{OK: false, Code: LandResultSourceFence, Preserved: true, Reason: reason, Detail: detail}
}

func landSourceSnapshot(root string, cfg landConfig, git GitRunner) (string, *Result) {
	if cfg.sourceLease == nil {
		return "", nil
	}
	f := cfg.sourceLease
	refuse := func(detail string) (string, *Result) {
		r := sourceFenceRefusal("source lease is not currently owned by the presented attempt", detail)
		return "", &r
	}
	if f.ID == "" || strings.TrimSpace(f.Holder) == "" || f.Generation <= 0 {
		return refuse("ID, holder and positive generation are required together")
	}
	store := leaseref.NewWithRunner(func(_ context.Context, dir string, args ...string) (string, int, error) {
		rc, out := run(git, dir, args)
		return out, rc, nil
	}, root)
	now := time.Now()
	if cfg.now != nil {
		now = cfg.now()
	}
	v, oid, err := store.FenceSnapshot(context.Background(), leaseref.Record{ID: f.ID, Holder: f.Holder, Generation: f.Generation}, now)
	if err != nil {
		return refuse(err.Error())
	}
	if !v.OK {
		return refuse(v.Reason + ": " + v.Detail)
	}
	if !sourceFenceObjectID(oid) {
		return refuse("source object ID is invalid")
	}
	return oid, nil
}

func sourceFenceObjectID(s string) bool {
	if len(s) != 40 && len(s) != 64 {
		return false
	}
	_, err := hex.DecodeString(s)
	return err == nil && strings.Trim(s, "0") != ""
}

// landCompareAndSwap uses Git's one ref transaction to lock and verify the
// source ref AND compare/update the trunk. A release, renewal or generation
// replacement cannot fit between verification and the trunk update. No callback
// check, process-local mutex or preflight-only lease check provides this fence.
func landCompareAndSwap(root, target, candidate, parent string, cfg landConfig, git GitRunner) (int, string, *Result) {
	if cfg.sourceLease == nil {
		rc, out := run(git, root, []string{"update-ref", target, candidate, parent})
		return rc, out, nil
	}
	oid, refusal := landSourceSnapshot(root, cfg, git)
	if refusal != nil {
		return 1, "", refusal
	}
	if !strings.HasPrefix(target, "refs/heads/") || !sourceFenceObjectID(candidate) || !sourceFenceObjectID(parent) {
		r := sourceFenceRefusal("invalid source-fenced trunk CAS identity", "")
		return 1, "", &r
	}
	if rc, _ := run(git, root, []string{"check-ref-format", target}); rc != 0 {
		r := sourceFenceRefusal("invalid source-fenced target ref", "")
		return 1, "", &r
	}
	ref := (leaseref.Record{ID: cfg.sourceLease.ID}).Ref()
	input := "start\noption no-deref\nverify " + ref + " " + oid + "\noption no-deref\nupdate " + target + " " + candidate + " " + parent + "\nprepare\n"
	transact := cfg.sourceCAS
	if transact == nil {
		transact = sourceFenceGitTransaction
	}
	rc, out := transact(root, input, func(ctx context.Context) error {
		// Git holds BOTH prepared ref locks here. Unlike a preflight read,
		// this direct-type check cannot race a native ref writer before commit.
		reader := git
		if reader == nil {
			reader = BoundedGitRunner(ctx)
		}
		current, refusal := landSourceSnapshot(root, cfg, reader)
		if refusal != nil {
			return fmt.Errorf("%s: %s", refusal.Reason, refusal.Detail)
		}
		if current != oid {
			return fmt.Errorf("prepared Source object changed")
		}
		if rc, _ := run(reader, root, []string{"symbolic-ref", "--quiet", target}); rc != 1 {
			return fmt.Errorf("prepared trunk target must be a direct ref")
		}
		return ctx.Err()
	})
	if rc != 0 {
		// A nonzero subprocess result is not proof that no mutation happened
		// (e.g. lost process/pipe after commit). Preserve the candidate and require
		// readback; never silently retry an uncertain transaction as a fresh owner.
		r := Result{OK: false, Code: LandResultSourceFenceContended, Preserved: true,
			Reason: "atomic source/trunk CAS did not report success; read back refs before retry", Detail: tail(out, 300)}
		return rc, out, &r
	}
	return 0, out, nil
}

// Git's prepared transaction holds the source and trunk ref locks until commit
// or abort. The callback is deliberately after prepare: no-deref alone still
// compares a symbolic ref's resolved OID and is NOT proof of a direct ref.
func sourceFenceGitTransaction(root, input string, beforeCommit func(context.Context) error) (int, string) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "git", "update-ref", "--stdin")
	cmd.Dir = root
	cmd.WaitDelay = 100 * time.Millisecond
	windowgate.ConfigureBackgroundCommand(cmd)
	in, err := cmd.StdinPipe()
	if err != nil {
		return 127, err.Error()
	}
	out, err := cmd.StdoutPipe()
	if err != nil {
		_ = in.Close()
		return 127, err.Error()
	}
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Start(); err != nil {
		_ = in.Close()
		_ = out.Close()
		return 127, err.Error()
	}
	type response struct {
		line string
		err  error
	}
	responses := make(chan response, 1)
	go func() {
		scan := bufio.NewScanner(out)
		for scan.Scan() {
			select {
			case responses <- response{line: scan.Text()}:
			case <-ctx.Done():
				return
			}
		}
		err := scan.Err()
		if err == nil {
			err = io.EOF
		}
		select {
		case responses <- response{err: err}:
		case <-ctx.Done():
		}
	}()
	var transcript strings.Builder
	ack := func(expected string) error {
		select {
		case r := <-responses:
			if r.err != nil {
				return r.err
			}
			transcript.WriteString(r.line + "\n")
			if r.line != expected {
				return fmt.Errorf("unexpected ref transaction response %q", r.line)
			}
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	finish := func(cause error) (int, string) {
		_ = in.Close()
		if cause != nil {
			cancel()
			_ = out.Close()
		}
		err := cmd.Wait()
		_ = out.Close()
		if cause != nil {
			return 1, cause.Error() + "\n" + transcript.String() + stderr.String()
		}
		if err != nil {
			if e, ok := err.(*exec.ExitError); ok {
				return e.ExitCode(), transcript.String() + stderr.String()
			}
			return 127, err.Error() + "\n" + transcript.String() + stderr.String()
		}
		return 0, transcript.String()
	}
	if _, err := io.WriteString(in, input); err != nil {
		return finish(err)
	}
	if err := ack("start: ok"); err != nil {
		return finish(err)
	}
	if err := ack("prepare: ok"); err != nil {
		return finish(err)
	}
	if beforeCommit == nil {
		return finish(fmt.Errorf("missing prepared-lock source validator"))
	}
	if err := beforeCommit(ctx); err != nil {
		// An explicit abort lets Git release its prepared locks normally.
		_, _ = io.WriteString(in, "abort\n")
		_ = ack("abort: ok")
		return finish(err)
	}
	if _, err := io.WriteString(in, "commit\n"); err != nil {
		return finish(err)
	}
	if err := ack("commit: ok"); err != nil {
		return finish(err)
	}
	return finish(nil)
}
