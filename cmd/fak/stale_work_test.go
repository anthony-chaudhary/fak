package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/stalework"
)

func TestStaleWorkJSONBoundCheckpointResumeWitness(t *testing.T) {
	root := t.TempDir()
	for _, path := range []string{"docs/a.md", "docs/b.md", "docs/c.md"} {
		q := filepath.Join(root, filepath.FromSlash(path))
		if err := os.MkdirAll(filepath.Dir(q), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(q, []byte("ordinary prose"), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	oldScanner := staleWorkScanner
	t.Cleanup(func() { staleWorkScanner = oldScanner })
	var runner stalework.Runner
	staleWorkScanner = func(ctx context.Context, opt stalework.Options) (stalework.Packet, error) {
		opt.Now = time.Unix(1800000000, 0)
		opt.Run = runner
		return stalework.Scan(ctx, opt)
	}

	var firstInspected []string
	runner = func(ctx context.Context, _ string, args ...string) ([]byte, error) {
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

	var partialOut, partialErr bytes.Buffer
	started := time.Now()
	code := runStaleWork([]string{"--root", root, "--budget", "25ms", "--json"}, &partialOut, &partialErr)
	elapsed := time.Since(started)
	if code != 0 {
		t.Fatalf("partial code=%d stderr=%s", code, partialErr.String())
	}
	if elapsed > 2*time.Second {
		t.Fatalf("bounded CLI took %v", elapsed)
	}
	if !json.Valid(partialOut.Bytes()) {
		t.Fatalf("bounded CLI did not return JSON: %s", partialOut.String())
	}
	var partial stalework.Packet
	if err := json.Unmarshal(partialOut.Bytes(), &partial); err != nil {
		t.Fatal(err)
	}
	if partial.Discovery.Status != stalework.DiscoveryPartial ||
		partial.Discovery.Reason != stalework.ReasonDiscoveryBudget ||
		partial.Checkpoint == nil || partial.Checkpoint.NextPath != "docs/b.md" {
		t.Fatalf("partial=%+v checkpoint=%+v", partial.Discovery, partial.Checkpoint)
	}
	if !reflect.DeepEqual(firstInspected, []string{"docs/a.md", "docs/b.md"}) {
		t.Fatalf("first inspected=%v", firstInspected)
	}

	checkpointPath := filepath.Join(root, "partial.json")
	if err := os.WriteFile(checkpointPath, partialOut.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	var resumedInspected []string
	runner = func(_ context.Context, _ string, args ...string) ([]byte, error) {
		switch {
		case args[0] == "rev-parse":
			return []byte("head\n"), nil
		case args[0] == "ls-files":
			return []byte("docs/c.md\ndocs/a.md\ndocs/b.md\n"), nil
		case args[0] == "log" && args[1] == "-1":
			resumedInspected = append(resumedInspected, args[len(args)-1])
			return []byte("base|1609459200\n"), nil
		default:
			return nil, nil
		}
	}
	var completeOut, completeErr bytes.Buffer
	code = runStaleWork([]string{"--root", root, "--resume", checkpointPath, "--budget", "1s", "--json"}, &completeOut, &completeErr)
	if code != 0 {
		t.Fatalf("resume code=%d stderr=%s", code, completeErr.String())
	}
	var complete stalework.Packet
	if err := json.Unmarshal(completeOut.Bytes(), &complete); err != nil {
		t.Fatalf("resume JSON: %v\n%s", err, completeOut.String())
	}
	if complete.Discovery.Status != stalework.DiscoveryComplete || complete.Checkpoint != nil {
		t.Fatalf("complete=%+v checkpoint=%+v", complete.Discovery, complete.Checkpoint)
	}
	if !reflect.DeepEqual(resumedInspected, []string{"docs/b.md", "docs/c.md"}) {
		t.Fatalf("resume inspected=%v, want no replay of docs/a.md", resumedInspected)
	}

	// The before half is the issue's captured control-point failure. The after
	// half is generated from this run's actual elapsed time and parsed JSON, so
	// `go test -v -run TestStaleWorkJSONBoundCheckpointResumeWitness ./cmd/fak`
	// emits a durable before/after runtime + JSON witness with the fix.
	witness := struct {
		Before struct {
			ElapsedMillis int64  `json:"elapsed_millis"`
			JSONStatus    string `json:"json_status"`
			Source        string `json:"source"`
		} `json:"before"`
		After struct {
			ElapsedMillis       int64  `json:"elapsed_millis"`
			PacketElapsedMillis int64  `json:"packet_elapsed_millis"`
			JSONStatus          string `json:"json_status"`
			DiscoveryStatus     string `json:"discovery_status"`
			Reason              string `json:"reason"`
			NextPath            string `json:"next_path"`
			ResumeStatus        string `json:"resume_status"`
		} `json:"after"`
	}{}
	witness.Before.ElapsedMillis = 124000
	witness.Before.JSONStatus = "NO_OUTPUT"
	witness.Before.Source = "GitHub issue #6711 observed failure, 2026-08-13"
	witness.After.ElapsedMillis = elapsed.Milliseconds()
	witness.After.PacketElapsedMillis = partial.Metrics.ElapsedMillis
	witness.After.JSONStatus = "VALID"
	witness.After.DiscoveryStatus = partial.Discovery.Status
	witness.After.Reason = partial.Discovery.Reason
	witness.After.NextPath = partial.Checkpoint.NextPath
	witness.After.ResumeStatus = complete.Discovery.Status
	raw, err := json.Marshal(witness)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("stale-work-before-after-witness=%s", raw)
}

func TestStaleWorkSmallRepositoryJSONRemainsCompleteAndDeterministic(t *testing.T) {
	root := t.TempDir()
	for path, body := range map[string]string{
		"docs/a.md": "ordinary prose",
		"docs/b.md": "<!-- Code generated; DO NOT EDIT. -->",
	} {
		q := filepath.Join(root, filepath.FromSlash(path))
		if err := os.MkdirAll(filepath.Dir(q), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(q, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	oldScanner := staleWorkScanner
	t.Cleanup(func() { staleWorkScanner = oldScanner })
	staleWorkScanner = func(ctx context.Context, opt stalework.Options) (stalework.Packet, error) {
		opt.Now = time.Unix(1800000000, 0)
		opt.Run = func(_ context.Context, _ string, args ...string) ([]byte, error) {
			switch {
			case args[0] == "rev-parse":
				return []byte("head\n"), nil
			case args[0] == "ls-files":
				return []byte("docs/b.md\ndocs/a.md\n"), nil
			case args[0] == "log" && args[1] == "-1":
				return []byte("base|1609459200\n"), nil
			default:
				return nil, nil
			}
		}
		return stalework.Scan(ctx, opt)
	}

	run := func() []byte {
		t.Helper()
		var stdout, stderr bytes.Buffer
		if code := runStaleWork([]string{"--root", root, "--budget", "1s", "--json"}, &stdout, &stderr); code != 0 {
			t.Fatalf("code=%d stderr=%s", code, stderr.String())
		}
		return append([]byte(nil), stdout.Bytes()...)
	}
	first, second := run(), run()
	if !bytes.Equal(first, second) {
		t.Fatalf("complete small-repository JSON changed:\nfirst=%s\nsecond=%s", first, second)
	}
	var packet stalework.Packet
	if err := json.Unmarshal(first, &packet); err != nil {
		t.Fatal(err)
	}
	if packet.Discovery.Status != stalework.DiscoveryComplete || packet.Checkpoint != nil ||
		packet.Metrics.FilesScanned != 2 || packet.Metrics.FilesTotal != 2 ||
		packet.Metrics.ElapsedMillis != 0 {
		t.Fatalf("packet=%+v", packet)
	}
}
