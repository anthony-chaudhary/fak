package stalework

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestPacketDeterministicBoundedAndIntegrated(t *testing.T) {
	d := t.TempDir()
	write := func(p, s string) {
		t.Helper()
		q := filepath.Join(d, filepath.FromSlash(p))
		if err := os.MkdirAll(filepath.Dir(q), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(q, []byte(s), 0644); err != nil {
			t.Fatal(err)
		}
	}
	write("docs/a.md", "# Operate\nUse `fak serve`; currently Go 1.26 is required.\n"+strings.Repeat("x", 500))
	write("docs/b.md", "# Operate too\nUse internal/gateway for requests.")
	write("docs/notes/old.md", "# Historical study\nOld but valid.")
	write("docs/generated.md", "<!-- Code generated; DO NOT EDIT. -->")
	write("docs/quiet.md", "No dependency and no version assertion.")
	r := fixtureRunner()
	o := Options{Root: d, Limit: 10, Now: time.Unix(1800000000, 0), Run: r, OpenDedupe: map[string]bool{"stale-work:docs/b.md": true}}
	p1, e := Scan(context.Background(), o)
	if e != nil {
		t.Fatal(e)
	}
	p2, e := Scan(context.Background(), o)
	if e != nil {
		t.Fatal(e)
	}
	if !reflect.DeepEqual(p1, p2) {
		t.Fatal("repeated scan differed")
	}
	if p1.Discovery.Status != DiscoveryComplete || p1.Checkpoint != nil {
		t.Fatalf("discovery=%+v checkpoint=%+v, want deterministic complete result", p1.Discovery, p1.Checkpoint)
	}
	if p1.Metrics.FilesScanned != 5 || p1.Metrics.FilesTotal != 5 || p1.Metrics.ElapsedMillis != 0 {
		t.Fatalf("metrics=%+v, want complete deterministic counts without runtime noise", p1.Metrics)
	}
	if len(p1.Candidates) != 1 || p1.Candidates[0].Path != "docs/a.md" {
		t.Fatalf("candidates: %#v", p1.Candidates)
	}
	if p1.Candidates[0].Score < 70 || len(p1.Candidates[0].Excerpt) > maxExcerpt {
		t.Fatalf("unbounded/weak candidate: %#v", p1.Candidates[0])
	}
	if p1.Candidates[0].Batch != "cmd/fak" {
		t.Fatalf("batch=%q", p1.Candidates[0].Batch)
	}
	if !hasExemption(p1, "historical note") || !hasExemption(p1, "generated or third-party artifact") || !hasExemption(p1, "already-open candidate") {
		t.Fatalf("exemptions: %#v", p1.Exemptions)
	}
	if len(p1.Abstentions) != 1 || p1.Abstentions[0].Path != "docs/quiet.md" {
		t.Fatalf("abstentions: %#v", p1.Abstentions)
	}
}
func TestAgeAloneAbstains(t *testing.T) {
	d := t.TempDir()
	_ = os.WriteFile(filepath.Join(d, "old.md"), []byte("ordinary old prose"), 0644)
	p, e := Scan(context.Background(), Options{Root: d, Paths: []string{"old.md"}, Now: time.Unix(1900000000, 0), Run: fixtureRunner()})
	if e != nil {
		t.Fatal(e)
	}
	if len(p.Candidates) != 0 || len(p.Abstentions) != 1 {
		t.Fatalf("age became verdict: %#v", p)
	}
}

func TestDiscoveryBudgetReturnsCheckpointAndResumeSkipsScannedMembers(t *testing.T) {
	d := t.TempDir()
	for _, path := range []string{"docs/a.md", "docs/b.md", "docs/c.md"} {
		q := filepath.Join(d, filepath.FromSlash(path))
		if err := os.MkdirAll(filepath.Dir(q), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(q, []byte("ordinary prose"), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	var firstInspected []string
	firstRunner := func(ctx context.Context, _ string, args ...string) ([]byte, error) {
		switch {
		case args[0] == "rev-parse":
			return []byte("head\n"), nil
		case args[0] == "ls-files":
			return []byte("docs/c.md\ndocs/a.md\ndocs/b.md\n"), nil
		case args[0] == "log" && args[1] == "-1":
			path := args[len(args)-1]
			firstInspected = append(firstInspected, path)
			if path == "docs/b.md" {
				<-ctx.Done()
				return nil, ctx.Err()
			}
			return []byte("base|1609459200\n"), nil
		default:
			return nil, nil
		}
	}
	started := time.Now()
	partial, err := Scan(context.Background(), Options{
		Root: d, Limit: 10, Now: time.Unix(1800000000, 0),
		Run: firstRunner, Budget: 25 * time.Millisecond,
	})
	elapsed := time.Since(started)
	if err != nil {
		t.Fatal(err)
	}
	if elapsed > 2*time.Second {
		t.Fatalf("bounded scan took %v", elapsed)
	}
	if partial.Discovery.Status != DiscoveryPartial || partial.Discovery.Reason != ReasonDiscoveryBudget {
		t.Fatalf("discovery=%+v, want typed budget partial", partial.Discovery)
	}
	if partial.Checkpoint == nil || partial.Checkpoint.Schema != CheckpointSchema ||
		partial.Checkpoint.NextIndex != 1 || partial.Checkpoint.NextPath != "docs/b.md" {
		t.Fatalf("checkpoint=%+v, want resume at the interrupted member", partial.Checkpoint)
	}
	if partial.Metrics.FilesScanned != 1 || partial.Metrics.FilesTotal != 3 ||
		partial.Metrics.BudgetMillis != 25 || partial.Metrics.ElapsedMillis <= 0 {
		t.Fatalf("partial metrics=%+v", partial.Metrics)
	}
	if !reflect.DeepEqual(firstInspected, []string{"docs/a.md", "docs/b.md"}) {
		t.Fatalf("first inspected=%v", firstInspected)
	}
	if raw, err := json.Marshal(partial); err != nil || !json.Valid(raw) {
		t.Fatalf("partial JSON invalid: err=%v raw=%s", err, raw)
	}

	var resumedInspected []string
	resumeRunner := func(_ context.Context, _ string, args ...string) ([]byte, error) {
		switch {
		case args[0] == "rev-parse":
			return []byte("head\n"), nil
		case args[0] == "ls-files":
			return []byte("docs/c.md\ndocs/a.md\ndocs/b.md\n"), nil
		case args[0] == "log" && args[1] == "-1":
			path := args[len(args)-1]
			resumedInspected = append(resumedInspected, path)
			if path == "docs/a.md" {
				return nil, errors.New("resume replayed an already-scanned member")
			}
			return []byte("base|1609459200\n"), nil
		default:
			return nil, nil
		}
	}
	complete, err := Scan(context.Background(), Options{
		Root: d, Limit: 10, Run: resumeRunner, Budget: time.Second, Resume: &partial,
	})
	if err != nil {
		t.Fatal(err)
	}
	if complete.Discovery.Status != DiscoveryComplete || complete.Checkpoint != nil {
		t.Fatalf("resumed discovery=%+v checkpoint=%+v", complete.Discovery, complete.Checkpoint)
	}
	if complete.Metrics.FilesScanned != 3 || complete.Metrics.FilesTotal != 3 {
		t.Fatalf("resumed metrics=%+v", complete.Metrics)
	}
	if !reflect.DeepEqual(resumedInspected, []string{"docs/b.md", "docs/c.md"}) {
		t.Fatalf("resume inspected=%v, want only uncheckpointed members", resumedInspected)
	}
	if len(complete.Abstentions) != 3 {
		t.Fatalf("complete abstentions=%#v", complete.Abstentions)
	}
	if got := []string{complete.Abstentions[0].Path, complete.Abstentions[1].Path, complete.Abstentions[2].Path}; !reflect.DeepEqual(got, []string{"docs/a.md", "docs/b.md", "docs/c.md"}) {
		t.Fatalf("complete paths=%v", got)
	}
}

func TestBoundWitnessCapturesBeforeAndResumableAfter(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "stale-work-bound-witness-2026-08-13.json"))
	if err != nil {
		t.Fatal(err)
	}
	var witness struct {
		Issue  int `json:"issue"`
		Before struct {
			ElapsedMillis int64  `json:"elapsed_millis"`
			JSONStatus    string `json:"json_status"`
		} `json:"before"`
		After struct {
			BudgetMillis    int64    `json:"budget_millis"`
			ElapsedMillis   int64    `json:"elapsed_millis"`
			JSONStatus      string   `json:"json_status"`
			DiscoveryStatus string   `json:"discovery_status"`
			Reason          string   `json:"reason"`
			NextPath        string   `json:"next_path"`
			ResumeStatus    string   `json:"resume_status"`
			ResumedMembers  []string `json:"resumed_members"`
		} `json:"after"`
		RepositoryAfter struct {
			BudgetMillis    int64  `json:"budget_millis"`
			ElapsedMillis   int64  `json:"elapsed_millis"`
			WallElapsed     int64  `json:"wall_elapsed_millis"`
			JSONStatus      string `json:"json_status"`
			DiscoveryStatus string `json:"discovery_status"`
			Reason          string `json:"reason"`
			FilesScanned    int    `json:"files_scanned"`
			FilesTotal      int    `json:"files_total"`
			NextIndex       int    `json:"next_index"`
			NextPath        string `json:"next_path"`
		} `json:"repository_after"`
		RepositoryResume struct {
			WallElapsed     int64  `json:"wall_elapsed_millis"`
			JSONStatus      string `json:"json_status"`
			PriorNextIndex  int    `json:"prior_next_index"`
			NextIndex       int    `json:"next_index"`
			AdvancedMembers int    `json:"advanced_members"`
			FilesScanned    int    `json:"files_scanned"`
			DiscoveryStatus string `json:"discovery_status"`
			Reason          string `json:"reason"`
		} `json:"repository_resume"`
	}
	if err := json.Unmarshal(raw, &witness); err != nil {
		t.Fatal(err)
	}
	if witness.Issue != 6711 || witness.Before.ElapsedMillis != 124000 || witness.Before.JSONStatus != "NO_OUTPUT" {
		t.Fatalf("before witness=%+v", witness)
	}
	if witness.After.JSONStatus != "VALID" || witness.After.DiscoveryStatus != DiscoveryPartial ||
		witness.After.Reason != ReasonDiscoveryBudget || witness.After.ResumeStatus != DiscoveryComplete ||
		witness.After.ElapsedMillis > 2000 || witness.After.ElapsedMillis < witness.After.BudgetMillis ||
		witness.After.NextPath != "docs/b.md" ||
		!reflect.DeepEqual(witness.After.ResumedMembers, []string{"docs/b.md", "docs/c.md"}) {
		t.Fatalf("after witness=%+v", witness.After)
	}
	if witness.RepositoryAfter.JSONStatus != "VALID" ||
		witness.RepositoryAfter.DiscoveryStatus != DiscoveryPartial ||
		witness.RepositoryAfter.Reason != ReasonDiscoveryBudget ||
		witness.RepositoryAfter.BudgetMillis != 60000 ||
		witness.RepositoryAfter.ElapsedMillis < witness.RepositoryAfter.BudgetMillis ||
		witness.RepositoryAfter.WallElapsed > 120000 ||
		witness.RepositoryAfter.FilesScanned != witness.RepositoryAfter.NextIndex ||
		witness.RepositoryAfter.FilesScanned <= 0 ||
		witness.RepositoryAfter.FilesScanned >= witness.RepositoryAfter.FilesTotal ||
		witness.RepositoryAfter.NextPath == "" {
		t.Fatalf("repository after witness=%+v", witness.RepositoryAfter)
	}
	if witness.RepositoryResume.JSONStatus != "VALID" ||
		witness.RepositoryResume.DiscoveryStatus != DiscoveryPartial ||
		witness.RepositoryResume.Reason != ReasonDiscoveryBudget ||
		witness.RepositoryResume.WallElapsed > 5000 ||
		witness.RepositoryResume.NextIndex <= witness.RepositoryResume.PriorNextIndex ||
		witness.RepositoryResume.AdvancedMembers != witness.RepositoryResume.NextIndex-witness.RepositoryResume.PriorNextIndex ||
		witness.RepositoryResume.FilesScanned != witness.RepositoryResume.NextIndex {
		t.Fatalf("repository resume witness=%+v", witness.RepositoryResume)
	}
}

func fixtureRunner() Runner {
	return func(_ context.Context, _ string, a ...string) ([]byte, error) {
		if a[0] == "rev-parse" {
			return []byte("head\n"), nil
		}
		if a[0] == "ls-files" {
			return []byte("docs/quiet.md\ndocs/generated.md\ndocs/b.md\ndocs/a.md\ndocs/notes/old.md\n"), nil
		}
		if a[0] == "log" && a[1] == "-1" {
			return []byte("base|1609459200\n"), nil
		}
		if a[0] == "log" && a[1] == "--format=%H|%s" {
			return []byte("new|semantic dependency change\n"), nil
		}
		return nil, nil
	}
}
func hasExemption(p Packet, s string) bool {
	for _, x := range p.Exemptions {
		if x.Exemption == s {
			return true
		}
	}
	return false
}
