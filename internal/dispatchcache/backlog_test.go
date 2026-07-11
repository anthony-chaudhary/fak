package dispatchcache

import (
	"encoding/json"
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
