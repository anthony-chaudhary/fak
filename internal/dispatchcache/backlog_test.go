package dispatchcache

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestMergeBacklogUpdatesAddsAndCloses(t *testing.T) {
	base := []BacklogIssue{{1, json.RawMessage(`{"v":"old"}`)}, {2, json.RawMessage(`{"v":"closed"}`)}}
	changed := []BacklogIssue{{1, json.RawMessage(`{"v":"new"}`)}, {3, json.RawMessage(`{"v":"added"}`)}}
	got := MergeBacklog(base, changed, []int{2})
	if len(got) != 2 || got[0].Number != 1 || string(got[0].Data) != `{"v":"new"}` || got[1].Number != 3 {
		t.Fatalf("got=%+v", got)
	}
}
func TestBacklogSnapshotPersistsWatermark(t *testing.T) {
	p := filepath.Join(t.TempDir(), "b.json")
	now := time.Unix(100, 0).UTC()
	if err := WriteBacklog(p, "k", now, []BacklogIssue{{1, json.RawMessage(`{}`)}}); err != nil {
		t.Fatal(err)
	}
	got, ok := ReadBacklog(p, "k")
	if !ok || !got.Watermark.Equal(now) || len(got.Issues) != 1 {
		t.Fatalf("got=%+v ok=%v", got, ok)
	}
}

// backlogFingerprint returns the bytes and mtime of the snapshot, which together witness whether
// a tick rewrote it.
func backlogFingerprint(t *testing.T, path string) ([]byte, time.Time) {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	st, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	return b, st.ModTime()
}

// A tick whose delta changed nothing must not rewrite the snapshot at all: same bytes, same
// mtime, across two consecutive quiet ticks -- while the watermark still advances (#6092).
func TestSyncBacklogQuietTickLeavesSnapshotUnwritten(t *testing.T) {
	p := filepath.Join(t.TempDir(), "backlog.json")
	base := []BacklogIssue{{1, json.RawMessage(`{"v":"a"}`)}, {2, json.RawMessage(`{"v":"b"}`)}}
	start := time.Unix(1000, 0).UTC()
	if err := WriteBacklog(p, "k", start, base); err != nil {
		t.Fatal(err)
	}
	// Backdate the snapshot so a rewrite is visible regardless of filesystem timestamp
	// granularity.
	old := time.Now().Add(-time.Hour)
	if err := os.Chtimes(p, old, old); err != nil {
		t.Fatal(err)
	}
	wantBytes, wantMtime := backlogFingerprint(t, p)

	for i, tick := range []time.Time{start.Add(time.Minute), start.Add(2 * time.Minute)} {
		merged, err := SyncBacklog(p, "k", tick, base, nil, nil)
		if err != nil {
			t.Fatalf("tick %d: %v", i, err)
		}
		if !sameBacklog(merged, base) {
			t.Fatalf("tick %d: merged=%+v want=%+v", i, merged, base)
		}
		gotBytes, gotMtime := backlogFingerprint(t, p)
		if string(gotBytes) != string(wantBytes) {
			t.Fatalf("tick %d: snapshot bytes rewritten\n before=%s\n  after=%s", i, wantBytes, gotBytes)
		}
		if !gotMtime.Equal(wantMtime) {
			t.Fatalf("tick %d: snapshot mtime moved %v -> %v", i, wantMtime, gotMtime)
		}
		snap, ok := ReadBacklog(p, "k")
		if !ok || !snap.Watermark.Equal(tick) {
			t.Fatalf("tick %d: watermark=%v ok=%v want=%v", i, snap.Watermark, ok, tick)
		}
		if len(snap.Issues) != len(base) {
			t.Fatalf("tick %d: issues=%d want %d", i, len(snap.Issues), len(base))
		}
	}
}

// A sidecar written under a different cache key must not leak into this key's watermark.
func TestSyncBacklogWatermarkSidecarIsKeyScoped(t *testing.T) {
	p := filepath.Join(t.TempDir(), "backlog.json")
	start := time.Unix(1000, 0).UTC()
	if err := WriteBacklog(p, "k", start, nil); err != nil {
		t.Fatal(err)
	}
	if err := WriteBacklogWatermark(p, "other", start.Add(time.Hour)); err != nil {
		t.Fatal(err)
	}
	snap, ok := ReadBacklog(p, "k")
	if !ok || !snap.Watermark.Equal(start) {
		t.Fatalf("watermark=%v ok=%v want=%v", snap.Watermark, ok, start)
	}
}

// A tick that learns something still rewrites the whole snapshot, and clears the sidecar the
// preceding quiet ticks left behind.
func TestSyncBacklogRewritesSnapshotWhenAnIssueChanges(t *testing.T) {
	for _, tc := range []struct {
		name    string
		changed []BacklogIssue
		closed  []int
		want    int
	}{
		{name: "updated", changed: []BacklogIssue{{1, json.RawMessage(`{"v":"new"}`)}}, want: 2},
		{name: "added", changed: []BacklogIssue{{3, json.RawMessage(`{"v":"c"}`)}}, want: 3},
		{name: "closed", closed: []int{2}, want: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := filepath.Join(t.TempDir(), "backlog.json")
			base := []BacklogIssue{{1, json.RawMessage(`{"v":"a"}`)}, {2, json.RawMessage(`{"v":"b"}`)}}
			start := time.Unix(1000, 0).UTC()
			if err := WriteBacklog(p, "k", start, base); err != nil {
				t.Fatal(err)
			}
			quiet := start.Add(time.Minute)
			if _, err := SyncBacklog(p, "k", quiet, base, nil, nil); err != nil {
				t.Fatal(err)
			}
			old := time.Now().Add(-time.Hour)
			if err := os.Chtimes(p, old, old); err != nil {
				t.Fatal(err)
			}
			_, staleMtime := backlogFingerprint(t, p)

			loud := start.Add(2 * time.Minute)
			merged, err := SyncBacklog(p, "k", loud, base, tc.changed, tc.closed)
			if err != nil {
				t.Fatal(err)
			}
			if len(merged) != tc.want {
				t.Fatalf("merged=%+v want %d rows", merged, tc.want)
			}
			if _, mtime := backlogFingerprint(t, p); mtime.Equal(staleMtime) {
				t.Fatalf("snapshot not rewritten on a changed delta (mtime still %v)", staleMtime)
			}
			snap, ok := ReadBacklog(p, "k")
			if !ok || !snap.Watermark.Equal(loud) || len(snap.Issues) != tc.want {
				t.Fatalf("snap=%+v ok=%v", snap, ok)
			}
			if !sameBacklog(snap.Issues, merged) {
				t.Fatalf("persisted=%+v merged=%+v", snap.Issues, merged)
			}
			if _, err := os.Stat(backlogWatermarkPath(p)); !os.IsNotExist(err) {
				t.Fatalf("stale watermark sidecar survived a full write: err=%v", err)
			}
		})
	}
}
