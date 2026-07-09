package stepbaton

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestNormalizeStepClassFailsClosed pins the closed vocabulary and the fail-closed
// default: every member survives, everything else folds to unknown (never to a
// confident "any").
func TestNormalizeStepClassFailsClosed(t *testing.T) {
	for _, ok := range []string{StepAny, StepBounded, StepCheckpoint, StepRebuild, StepUnknown} {
		if got := NormalizeStepClass(ok); got != ok {
			t.Errorf("NormalizeStepClass(%q) = %q, want unchanged", ok, got)
		}
		if !ValidStepClass(ok) {
			t.Errorf("ValidStepClass(%q) = false, want true", ok)
		}
	}
	for _, bad := range []string{"", "  ", "ANY", "huge", "critical", "../rebuild"} {
		if got := NormalizeStepClass(bad); got != StepUnknown {
			t.Errorf("NormalizeStepClass(%q) = %q, want %q", bad, got, StepUnknown)
		}
		if ValidStepClass(bad) {
			t.Errorf("ValidStepClass(%q) = true, want false", bad)
		}
	}
	// A whitespace-padded member is invalid as a literal but normalizes to the member:
	// Normalize trims, so padding is forgiven rather than folded to unknown.
	if got := NormalizeStepClass("  checkpoint  "); got != StepCheckpoint {
		t.Errorf("NormalizeStepClass(padded) = %q, want %q", got, StepCheckpoint)
	}
	if ValidStepClass("  checkpoint  ") {
		t.Errorf("ValidStepClass(padded) = true, want false (literal, untrimmed)")
	}
}

// TestNewStampsSchemaAndNormalizes proves New stamps the schema tag, trims lineage
// fields, and normalizes the class fail-closed in one place.
func TestNewStampsSchemaAndNormalizes(t *testing.T) {
	s := New("  trace-1 ", "CHECKPOINT-not-a-member", "token_headroom", "window nearly spent", "guard", "abc123", 92831, 48000)
	if s.Schema != Schema {
		t.Fatalf("Schema = %q, want %q", s.Schema, Schema)
	}
	if s.TraceID != "trace-1" {
		t.Errorf("TraceID = %q, want trimmed %q", s.TraceID, "trace-1")
	}
	if s.StepClass != StepUnknown {
		t.Errorf("StepClass = %q, want fail-closed %q", s.StepClass, StepUnknown)
	}
	if s.ResidentTokens != 92831 || s.BudgetTokens != 48000 {
		t.Errorf("tokens = (%d,%d), want (92831,48000)", s.ResidentTokens, s.BudgetTokens)
	}

	valid := New("t", StepCheckpoint, "token_headroom", "r", "guard", "", 100, 200)
	if valid.StepClass != StepCheckpoint {
		t.Errorf("valid StepClass = %q, want %q", valid.StepClass, StepCheckpoint)
	}
}

// TestMarshalUnmarshalRoundTrip proves the durable bytes round-trip and carry the
// schema tag.
func TestMarshalUnmarshalRoundTrip(t *testing.T) {
	s := New("t1", StepRebuild, "context_event", "event just fired", "guard", "deadbeef", 5000, 48000)
	data, err := Marshal(s)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if !strings.HasSuffix(string(data), "\n") {
		t.Errorf("Marshal output missing trailing newline")
	}
	if !strings.Contains(string(data), `"schema": "`+Schema+`"`) {
		t.Errorf("Marshal output missing schema tag:\n%s", data)
	}
	got, err := Unmarshal(data)
	if err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if got != s {
		t.Errorf("round trip mismatch:\n got %+v\nwant %+v", got, s)
	}
}

// TestShouldCarry pins the carry policy: only the two classes that change successor
// behavior are worth injecting; the rest are noise.
func TestShouldCarry(t *testing.T) {
	carry := map[string]bool{
		StepAny:        false,
		StepBounded:    false,
		StepCheckpoint: true,
		StepRebuild:    true,
		StepUnknown:    false,
	}
	for class, want := range carry {
		s := New("t", class, "b", "r", "guard", "", 1, 2)
		if got := s.ShouldCarry(); got != want {
			t.Errorf("ShouldCarry(%q) = %v, want %v", class, got, want)
		}
	}
}

// TestLineRendersDecision proves the injected line leads with the decision and omits
// empty optional fields.
func TestLineRendersDecision(t *testing.T) {
	full := New("t", StepCheckpoint, "token_headroom", "nearly spent", "guard", "abc123", 92831, 48000)
	line := full.Line()
	for _, want := range []string{"last live step=checkpoint", "basis=token_headroom", "resident=92831 budget=48000", "phase=guard", "at=abc123", `reason="nearly spent"`} {
		if !strings.Contains(line, want) {
			t.Errorf("Line() = %q, missing %q", line, want)
		}
	}
	// Empty optionals drop out entirely rather than rendering blanks.
	sparse := New("", StepRebuild, "", "", "", "", 0, 0)
	sl := sparse.Line()
	if strings.Contains(sl, "basis=") || strings.Contains(sl, "resident=") || strings.Contains(sl, "phase=") || strings.Contains(sl, "at=") || strings.Contains(sl, "reason=") {
		t.Errorf("sparse Line() leaked an empty field: %q", sl)
	}
	if !strings.Contains(sl, "last live step=rebuild") {
		t.Errorf("sparse Line() = %q, want the decision", sl)
	}
}

// TestPathPerSessionAndSanitizes proves the path is per-session and that a hostile id
// cannot escape the directory via a separator or "..".
func TestPathPerSessionAndSanitizes(t *testing.T) {
	dir := "/var/run/fak"
	p := Path(dir, "sess-42")
	if want := filepath.Join(dir, "stepadvice-sess-42.json"); p != want {
		t.Errorf("Path = %q, want %q", p, want)
	}
	// Two different sessions get two different files.
	if Path(dir, "a") == Path(dir, "b") {
		t.Errorf("Path collided across distinct session ids")
	}
	for _, hostile := range []string{"../../etc/passwd", "a/b", "..", ".", `c:\windows`, ""} {
		got := Path(dir, hostile)
		base := filepath.Base(got)
		if strings.ContainsAny(strings.TrimPrefix(strings.TrimSuffix(base, ".json"), "stepadvice-"), `/\`) {
			t.Errorf("Path(%q) leaked a separator: %q", hostile, got)
		}
		if filepath.Dir(got) != filepath.Clean(dir) {
			t.Errorf("Path(%q) = %q escaped dir %q", hostile, got, dir)
		}
	}
}

// TestWriteReadRoundTripAndReplace proves Write→Read returns the stamp, that Write over
// an existing stamp replaces it (atomic swap), and that the parent dir is created.
func TestWriteReadRoundTripAndReplace(t *testing.T) {
	dir := t.TempDir()
	path := Path(filepath.Join(dir, "nested"), "sess-1") // nested dir must be created

	first := New("t1", StepCheckpoint, "token_headroom", "one", "guard", "sha1", 90000, 48000)
	if err := Write(path, first); err != nil {
		t.Fatalf("Write first: %v", err)
	}
	got, present, err := Read(path)
	if err != nil || !present {
		t.Fatalf("Read after write: present=%v err=%v", present, err)
	}
	if got != first {
		t.Errorf("read mismatch:\n got %+v\nwant %+v", got, first)
	}

	second := New("t2", StepRebuild, "context_event", "two", "guard", "sha2", 5000, 48000)
	if err := Write(path, second); err != nil {
		t.Fatalf("Write second: %v", err)
	}
	got2, _, err := Read(path)
	if err != nil {
		t.Fatalf("Read after replace: %v", err)
	}
	if got2 != second {
		t.Errorf("replace did not take:\n got %+v\nwant %+v", got2, second)
	}
	// No stray temp files left behind by the atomic writer.
	entries, _ := os.ReadDir(filepath.Dir(path))
	for _, e := range entries {
		if strings.Contains(e.Name(), ".tmp-") {
			t.Errorf("atomic Write left a temp file: %s", e.Name())
		}
	}
}

// TestReadAbsentIsDecidable proves an absent file is (zero,false,nil) — a resume with
// no carryover injects nothing rather than erroring.
func TestReadAbsentIsDecidable(t *testing.T) {
	dir := t.TempDir()
	s, present, err := Read(Path(dir, "never-written"))
	if err != nil {
		t.Fatalf("Read absent returned error: %v", err)
	}
	if present {
		t.Errorf("present = true for an absent file")
	}
	if s != (Stamp{}) {
		t.Errorf("absent Read returned a non-zero stamp: %+v", s)
	}
}

// TestReadCorruptIsError proves a present-but-garbage file is surfaced as an error, not
// silently treated as "no carryover".
func TestReadCorruptIsError(t *testing.T) {
	dir := t.TempDir()
	path := Path(dir, "corrupt")
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("{not json"), 0o600); err != nil {
		t.Fatal(err)
	}
	_, present, err := Read(path)
	if err == nil {
		t.Errorf("Read of corrupt file returned nil error")
	}
	if present {
		t.Errorf("present = true for a corrupt file")
	}
}
