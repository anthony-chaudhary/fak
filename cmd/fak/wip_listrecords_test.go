package main

import (
	"context"
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/wipref"
)

// TestWipListRecordsSingleCall proves the folded listing path: wipListRecords pulls
// every checkpoint's object AND stamp in one `git for-each-ref` (no git-log-per-ref
// fan-out), correctly decoding the stamp even when the marker is NOT the commit's
// first line — a multi-line message, the case a subject-only read would miss — and
// falling back to the ref-name label when a ref carries no parseable stamp.
func TestWipListRecordsSingleCall(t *testing.T) {
	ctx := context.Background()
	dir, file := wipTestRepo(t)

	// Two real checkpoints from evolving working-tree deltas.
	if err := os.WriteFile(file, []byte("base line\nedit A\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	ra, err := wipCheckpoint(ctx, dir, "sess-a", true, 1000)
	if err != nil || ra.Object == "" {
		t.Fatalf("checkpoint sess-a: %v (%+v)", err, ra)
	}
	if err := os.WriteFile(file, []byte("base line\nedit A\nedit B\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	rb, err := wipCheckpoint(ctx, dir, "sess-b", false, 2000)
	if err != nil || rb.Object == "" {
		t.Fatalf("checkpoint sess-b: %v (%+v)", err, rb)
	}

	head, err := gitWipOut(ctx, dir, nil, "rev-parse", "HEAD")
	if err != nil {
		t.Fatal(err)
	}
	tree, err := gitWipOut(ctx, dir, nil, "rev-parse", "HEAD^{tree}")
	if err != nil {
		t.Fatal(err)
	}

	// sess-c: a multi-line message whose stamp line is NOT the subject. This is the
	// case that discriminates full-contents parsing (correct) from a subject-only read.
	stampC, err := wipref.EncodeStamp(wipref.Stamp{
		SessionID: "sess-c", StartSHA: head, Buildable: true, CheckpointedAt: 3000,
	})
	if err != nil {
		t.Fatal(err)
	}
	msgC := "a human prose subject a hook might prepend\n\n" + stampC + "\n"
	commitC, err := gitWipOut(ctx, dir, nil, "commit-tree", tree, "-p", head, "-m", msgC)
	if err != nil {
		t.Fatalf("commit-tree sess-c: %v", err)
	}
	if _, err := gitWipOut(ctx, dir, nil, "update-ref", wipref.SessionRef("sess-c"), commitC); err != nil {
		t.Fatal(err)
	}

	// sess-d: no parseable stamp at all — must fall back to the ref-name label.
	commitD, err := gitWipOut(ctx, dir, nil, "commit-tree", tree, "-p", head, "-m", "no stamp here")
	if err != nil {
		t.Fatalf("commit-tree sess-d: %v", err)
	}
	if _, err := gitWipOut(ctx, dir, nil, "update-ref", wipref.SessionRef("sess-d"), commitD); err != nil {
		t.Fatal(err)
	}

	recs, err := wipListRecords(ctx, dir)
	if err != nil {
		t.Fatalf("wipListRecords: %v", err)
	}
	got := map[string]wipref.RefRecord{}
	for _, r := range recs {
		got[wipSessionOf(r)] = r
	}
	if len(recs) != 4 {
		t.Fatalf("want 4 records, got %d: sessions=%v", len(recs), mapKeys(got))
	}
	// sess-a / sess-b: real stamps decoded, objects match the checkpoints, buildable preserved.
	if a := got["sess-a"]; a.Object != ra.Object || a.Stamp.CheckpointedAt != 1000 || !a.Stamp.Buildable {
		t.Errorf("sess-a mismatch: %+v (want obj %s, ts 1000, buildable true)", a, ra.Object)
	}
	if b := got["sess-b"]; b.Object != rb.Object || b.Stamp.Buildable {
		t.Errorf("sess-b mismatch: %+v (want obj %s, buildable false)", b, rb.Object)
	}
	// sess-c: stamp decoded from a NON-first line — the multi-line case.
	if c := got["sess-c"]; c.Object != commitC || c.Stamp.SessionID != "sess-c" || c.Stamp.CheckpointedAt != 3000 {
		t.Errorf("sess-c multi-line stamp not decoded: %+v", c)
	}
	// sess-d: no stamp -> session label recovered from the ref name, empty StartSHA.
	if d := got["sess-d"]; d.Object != commitD || d.Stamp.SessionID != "sess-d" || d.Stamp.StartSHA != "" {
		t.Errorf("sess-d fallback wrong: %+v", d)
	}
}

func mapKeys(m map[string]wipref.RefRecord) []string {
	ks := make([]string, 0, len(m))
	for k := range m {
		ks = append(ks, k)
	}
	return ks
}
