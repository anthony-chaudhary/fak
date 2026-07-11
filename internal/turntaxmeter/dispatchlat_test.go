package turntaxmeter

import (
	"reflect"
	"testing"
)

func TestFoldDispatchLatencyGolden(t *testing.T) {
	rows := []map[string]int64{}
	for i := int64(1); i <= 100; i++ {
		rows = append(rows, map[string]int64{"preflight": i, "total": i * 2})
	}
	got := FoldDispatchLatency(rows)
	want := []DispatchLatency{{"preflight", 100, 50, 90, 99}, {"total", 100, 100, 180, 198}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got=%+v want=%+v", got, want)
	}
}
