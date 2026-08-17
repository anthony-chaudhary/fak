package toolcallcontrol

import "testing"

func TestReplayCompactBucketsReasonsAndRecordLinks(t *testing.T) {
	rows := []ReplayRow{
		{ID: "short", Turn: 1, Tool: "read_file", Args: []byte(`{"p":"a"}`), ReadOnly: true, StateEpoch: "s", PromptUnits: 7000, Needed: boolp(true), ResultID: "r", Succeeded: true},
		{ID: "long", Turn: 2, Tool: "read_file", Args: []byte(`{"p":"a"}`), ReadOnly: true, StateEpoch: "s", PromptUnits: 128000, Needed: boolp(false), ResultID: "r2", Succeeded: true},
	}
	compact := Replay(rows).Compact()
	var exact ReplayCompactArm
	for _, arm := range compact.Arms {
		if arm.Name == "exact-reuse" {
			exact = arm
		}
	}
	if exact.Records == "" || len(exact.Buckets) != 2 || len(exact.Reasons) == 0 {
		t.Fatalf("compact=%+v", exact)
	}
	for _, bucket := range exact.Buckets {
		if bucket.Name == "gte-128k" && (bucket.UnneededAvoided != 1 || bucket.NeededSuppressed != 0) {
			t.Fatalf("bucket=%+v", bucket)
		}
	}
}
