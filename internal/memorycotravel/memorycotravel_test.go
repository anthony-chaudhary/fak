package memorycotravel

import (
	"os"
	"path/filepath"
	"sort"
	"testing"
	"time"
)

const (
	testSID  = "deadbeef-0000-1111-2222-333344445555"
	testSlug = "C--work-fak"
)

func memDir(t *testing.T, cfg string, slug ...string) string {
	t.Helper()
	s := testSlug
	if len(slug) > 0 {
		s = slug[0]
	}
	d := filepath.Join(cfg, "projects", s, "memory")
	if err := os.MkdirAll(d, 0o755); err != nil {
		t.Fatal(err)
	}
	return d
}

func write(t *testing.T, path, text string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(text), 0o644); err != nil {
		t.Fatal(err)
	}
}

func ledgerInto(t *testing.T, dir string) string {
	t.Helper()
	p := filepath.Join(dir, "cotravel-ledger.jsonl")
	t.Setenv("FAK_MEMORY_COTRAVEL_LEDGER", p)
	return p
}

func TestStrategies(t *testing.T) {
	src := memDir(t, filepath.Join(t.TempDir(), "A"))
	dst := memDir(t, filepath.Join(t.TempDir(), "B"))
	s := filepath.Join(src, "note.md")
	d := filepath.Join(dst, "note.md")
	write(t, s, "from-A")
	if Additive(s, d) != "copy" {
		t.Fatalf("additive should copy missing")
	}
	write(t, d, "from-B")
	if Additive(s, d) != "skip" {
		t.Fatalf("additive should skip existing")
	}
	if SourceWins(s, d) != "copy" {
		t.Fatalf("source wins should copy differing")
	}
	write(t, d, "from-A")
	if SourceWins(s, d) != "skip" {
		t.Fatalf("source wins should skip identical")
	}
	write(t, d, "from-B")
	if err := os.Chtimes(s, ts(3_000_000), ts(3_000_000)); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(d, ts(1_000_000), ts(1_000_000)); err != nil {
		t.Fatal(err)
	}
	if NewestMtime(s, d) != "copy" {
		t.Fatalf("newest should copy newer source")
	}
	if err := os.Chtimes(s, ts(1_000_000), ts(1_000_000)); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(d, ts(3_000_000), ts(3_000_000)); err != nil {
		t.Fatal(err)
	}
	if NewestMtime(s, d) != "skip" {
		t.Fatalf("newest should skip older source")
	}
}

func TestCotravelGates(t *testing.T) {
	t.Setenv("FAK_MEMORY_COTRAVEL", "")
	t.Setenv("FAK_MEMORY_MERGE", "")
	ledgerInto(t, t.TempDir())
	a, b := filepath.Join(t.TempDir(), "A"), filepath.Join(t.TempDir(), "B")
	src := memDir(t, a)
	dst := memDir(t, b)
	write(t, filepath.Join(src, "fresh.md"), "carry-me")
	write(t, filepath.Join(src, "conflict.md"), "A-version")
	write(t, filepath.Join(dst, "conflict.md"), "B-version")
	write(t, filepath.Join(dst, "dest-only.md"), "B-private")
	rec := CotravelMemory(a, b, testSlug, testSID, Options{Gate: "live", Strategy: "additive"})
	if got := sorted(rec.Copied); len(got) != 1 || got[0] != "fresh.md" {
		t.Fatalf("copied = %v", rec.Copied)
	}
	if data, _ := os.ReadFile(filepath.Join(dst, "conflict.md")); string(data) != "B-version" {
		t.Fatalf("conflict clobbered")
	}
	if _, err := os.Stat(filepath.Join(dst, "dest-only.md")); err != nil {
		t.Fatalf("dest-only pruned")
	}

	tmp := t.TempDir()
	ledger := ledgerInto(t, tmp)
	a, b = filepath.Join(tmp, "A"), filepath.Join(tmp, "B")
	src = memDir(t, a)
	_ = memDir(t, b)
	write(t, filepath.Join(src, "fresh.md"), "carry-me")
	rec = CotravelMemory(a, b, testSlug, testSID, Options{Gate: "shadow", Strategy: "additive"})
	if len(rec.Copied) != 0 || len(rec.WouldCopy) != 1 || rec.WouldCopy[0] != "fresh.md" {
		t.Fatalf("shadow record = %+v", rec)
	}
	if _, err := os.Stat(filepath.Join(b, "projects", testSlug, "memory", "fresh.md")); !os.IsNotExist(err) {
		t.Fatalf("shadow copied file")
	}
	if _, err := os.Stat(ledger); err != nil {
		t.Fatalf("ledger not written: %v", err)
	}
	rows := ReadLedger()
	if len(rows) == 0 || rows[len(rows)-1]["session"] != testSID || rows[len(rows)-1]["gate"] != "shadow" {
		t.Fatalf("bad ledger rows: %+v", rows)
	}

	tmp = t.TempDir()
	ledger = ledgerInto(t, tmp)
	a, b = filepath.Join(tmp, "A"), filepath.Join(tmp, "B")
	src = memDir(t, a)
	_ = memDir(t, b)
	write(t, filepath.Join(src, "fresh.md"), "carry-me")
	rec = CotravelMemory(a, b, testSlug, testSID, Options{Gate: "off"})
	if len(rec.Copied) != 0 || len(rec.Plan) != 0 {
		t.Fatalf("off should no-op: %+v", rec)
	}
	if _, err := os.Stat(ledger); !os.IsNotExist(err) {
		t.Fatalf("off wrote ledger")
	}
}

func TestDstSlugAndDefaults(t *testing.T) {
	tmp := t.TempDir()
	ledgerInto(t, tmp)
	a, b := filepath.Join(tmp, "A"), filepath.Join(tmp, "B")
	src := memDir(t, a, testSlug)
	write(t, filepath.Join(src, "note.md"), "owner-memory")
	other := "C--work-slack-helpers"
	rec := CotravelMemory(a, b, testSlug, testSID, Options{DstSlug: other, Gate: "live", Strategy: "additive"})
	if rec.DstSlug != other {
		t.Fatalf("dst slug = %s", rec.DstSlug)
	}
	if data, err := os.ReadFile(filepath.Join(b, "projects", other, "memory", "note.md")); err != nil || string(data) != "owner-memory" {
		t.Fatalf("memory did not land under dst slug: %q %v", data, err)
	}
	t.Setenv("FAK_MEMORY_COTRAVEL", "bogus")
	t.Setenv("FAK_MEMORY_MERGE", "bogus")
	if Gate() != "shadow" || StrategyName() != "additive" {
		t.Fatalf("bad defaults: %s/%s", Gate(), StrategyName())
	}
	tmp = t.TempDir()
	ledgerInto(t, tmp)
	a, b = filepath.Join(tmp, "A"), filepath.Join(tmp, "B")
	if err := os.MkdirAll(filepath.Join(a, "projects", testSlug), 0o755); err != nil {
		t.Fatal(err)
	}
	rec = CotravelMemory(a, b, testSlug, testSID, Options{Gate: "live"})
	if rec.SrcHasMemory || len(rec.Plan) != 0 || len(rec.Copied) != 0 {
		t.Fatalf("no-source-memory should be clean: %+v", rec)
	}
}

func sorted(items []string) []string {
	out := append([]string(nil), items...)
	sort.Strings(out)
	return out
}

func ts(sec int64) time.Time { return time.Unix(sec, 0) }
