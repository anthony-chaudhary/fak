package hooks

import (
	"strings"
	"testing"
)

// gate_gofmt_test.go — unit tests for the GOFMT gate. The gate reads d.StagedPaths and
// each path's bytes via d.FileBytes, which consults d.fileCache first — so these drive it
// over a hand-built StagedDiff with the file content injected into the cache, no git/disk.

// gofmtCleanSrc is canonical gofmt output; gofmtDirtySrc is the same program with the
// spacing gofmt normalizes away — the drift class the real gate caught on the trunk (a
// struct field widened without re-running gofmt collapses to exactly this: extra
// intra-statement spacing that format.Source rewrites).
const gofmtCleanSrc = "package x\n\nvar A = 1\n"
const gofmtDirtySrc = "package x\n\nvar  A  =  1\n"

func gofmtDiff(paths []string, files map[string]string) *StagedDiff {
	cache := map[string]fileEntry{}
	for p, c := range files {
		cache[p] = fileEntry{data: []byte(c), exists: true}
	}
	return &StagedDiff{StagedPaths: paths, fileCache: cache}
}

func TestGateGofmt_UnformattedGoFires(t *testing.T) {
	d := gofmtDiff(
		[]string{"internal/x/bad.go", "internal/x/good.go"},
		map[string]string{
			"internal/x/bad.go":  gofmtDirtySrc,
			"internal/x/good.go": gofmtCleanSrc,
		},
	)
	findings, err := gateGofmt(d)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(findings) != 1 {
		t.Fatalf("an unformatted staged .go should fire exactly one finding, got %d: %+v", len(findings), findings)
	}
	f := findings[0]
	if f.Gate != "GOFMT" {
		t.Fatalf("gate name = %q, want GOFMT", f.Gate)
	}
	if f.File != "internal/x/bad.go" {
		t.Fatalf("finding should anchor to the unformatted file, got %q", f.File)
	}
	if strings.Contains(f.Detail, "good.go") {
		t.Fatalf("a gofmt-clean file must not be named; got %q", f.Detail)
	}
	for _, want := range []string{"internal/x/bad.go", "gofmt -w", "1 staged .go file"} {
		if !strings.Contains(f.Detail, want) {
			t.Fatalf("detail should mention %q; got %q", want, f.Detail)
		}
	}
}

func TestGateGofmt_AllFormattedClean(t *testing.T) {
	d := gofmtDiff(
		[]string{"internal/x/good.go"},
		map[string]string{"internal/x/good.go": gofmtCleanSrc},
	)
	if findings, err := gateGofmt(d); err != nil || len(findings) != 0 {
		t.Fatalf("a gofmt-clean staged set must not fire; got findings=%+v err=%v", findings, err)
	}
}

func TestGateGofmt_NonGoIgnored(t *testing.T) {
	// A non-.go path whose bytes are not gofmt-clean is irrelevant — the gate scopes to .go.
	d := gofmtDiff(
		[]string{"docs/notes.md", "config.yaml"},
		map[string]string{"docs/notes.md": gofmtDirtySrc, "config.yaml": "a:  b\n"},
	)
	if findings, err := gateGofmt(d); err != nil || len(findings) != 0 {
		t.Fatalf("non-.go paths must be ignored; got findings=%+v err=%v", findings, err)
	}
}

func TestGateGofmt_UnparseableSkipped(t *testing.T) {
	// A .go file gofmt cannot parse is skipped (the build/vet gate owns syntax), so the gate
	// stays clean rather than mis-flagging it.
	d := gofmtDiff(
		[]string{"internal/x/broken.go"},
		map[string]string{"internal/x/broken.go": "package x\nfunc ( bad syntax {{{\n"},
	)
	if findings, err := gateGofmt(d); err != nil || len(findings) != 0 {
		t.Fatalf("an unparseable .go must be skipped, not flagged; got findings=%+v err=%v", findings, err)
	}
}

func TestGateGofmt_EmptyStagedSetClean(t *testing.T) {
	if findings, err := gateGofmt(&StagedDiff{}); err != nil || len(findings) != 0 {
		t.Fatalf("an empty staged set fires nothing; got findings=%+v err=%v", findings, err)
	}
}

func TestGateGofmt_DefaultModeBlocks(t *testing.T) {
	for _, gate := range PreCommitGates() {
		if gate.Name != "GOFMT" {
			continue
		}
		if gate.DefaultMode != "" {
			t.Fatalf("GOFMT DefaultMode=%q, want empty (block)", gate.DefaultMode)
		}
		if gate.ModeEnv != "FLEET_GOFMT_GUARD" || gate.EscapeEnv != "ALLOW_GOFMT_DRIFT" {
			t.Fatalf("GOFMT gate modes changed unexpectedly: %+v", gate)
		}
		return
	}
	t.Fatal("GOFMT gate not registered")
}
