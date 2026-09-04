package wipinventory

import (
	"fmt"
	"strings"
	"testing"
	"time"
)

type reapFakeRunner struct {
	calls []string
	refs  string
}

func (r *reapFakeRunner) Run(dir string, args ...string) ([]byte, error) {
	cmd := strings.Join(args, " ")
	r.calls = append(r.calls, cmd)
	if strings.HasPrefix(cmd, "for-each-ref") {
		return []byte(r.refs), nil
	}
	if strings.HasPrefix(cmd, "update-ref -d") {
		return nil, nil
	}
	return nil, fmt.Errorf("unexpected: %s", cmd)
}

func TestReapDeletesStaleRefs(t *testing.T) {
	now := time.Now()
	staleTime := now.Add(-8 * 24 * time.Hour).Unix()
	freshTime := now.Add(-1 * time.Hour).Unix()

	runner := &reapFakeRunner{
		refs: fmt.Sprintf("refs/fak/wip/stale\x00sha1\x00%d\nrefs/fak/wip/fresh\x00sha2\x00%d\n", staleTime, freshTime),
	}

	res, err := Reap("/test/repo", 7*24*time.Hour, false, runner)
	if err != nil {
		t.Fatalf("Reap error: %v", err)
	}

	if len(res.Reaped) != 1 || res.Reaped[0] != "refs/fak/wip/stale" {
		t.Fatalf("expected 1 reaped ref, got: %#v", res.Reaped)
	}
	if len(res.Kept) != 1 || res.Kept[0] != "refs/fak/wip/fresh" {
		t.Fatalf("expected 1 kept ref, got: %#v", res.Kept)
	}

	foundDelete := false
	for _, call := range runner.calls {
		if call == "update-ref -d refs/fak/wip/stale" {
			foundDelete = true
		}
		if call == "update-ref -d refs/fak/wip/fresh" {
			t.Fatalf("unexpected delete of fresh ref: %s", call)
		}
	}
	if !foundDelete {
		t.Fatalf("missing expected delete call, got calls: %v", runner.calls)
	}
}

func TestReapDryRunDoesNotDelete(t *testing.T) {
	now := time.Now()
	staleTime := now.Add(-10 * 24 * time.Hour).Unix()

	runner := &reapFakeRunner{
		refs: fmt.Sprintf("refs/fak/wip/stale\x00sha1\x00%d\n", staleTime),
	}

	res, err := Reap("/test/repo", 7*24*time.Hour, true, runner)
	if err != nil {
		t.Fatalf("Reap error: %v", err)
	}

	if len(res.Reaped) != 1 || res.Reaped[0] != "refs/fak/wip/stale" {
		t.Fatalf("expected stale in reaped list for dry-run: %#v", res.Reaped)
	}
	for _, call := range runner.calls {
		if strings.HasPrefix(call, "update-ref") {
			t.Fatalf("dry-run should not call update-ref, got: %s", call)
		}
	}
}

func TestReapDefaultMaxAge(t *testing.T) {
	now := time.Now()
	staleTime := now.Add(-8 * 24 * time.Hour).Unix()
	freshTime := now.Add(-6 * 24 * time.Hour).Unix()

	runner := &reapFakeRunner{
		refs: fmt.Sprintf("refs/fak/wip/stale\x00sha1\x00%d\nrefs/fak/wip/fresh\x00sha2\x00%d\n", staleTime, freshTime),
	}

	// 0 should default to 7 days
	res, err := Reap("/test/repo", 0, false, runner)
	if err != nil {
		t.Fatalf("Reap error: %v", err)
	}

	if len(res.Reaped) != 1 || res.Reaped[0] != "refs/fak/wip/stale" {
		t.Fatalf("expected 1 reaped ref with default maxAge: %#v", res.Reaped)
	}
	if len(res.Kept) != 1 || res.Kept[0] != "refs/fak/wip/fresh" {
		t.Fatalf("expected 1 kept ref with default maxAge: %#v", res.Kept)
	}
}
