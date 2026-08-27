package model

import "testing"

func TestProveOperationOverlap(t *testing.T) {
	ops := []OverlapOperation{{ID: "copy", Kind: "copy", StartNS: 0, EndNS: 100, Bytes: 4096}, {ID: "compute-independent", Kind: "compute", StartNS: 20, EndNS: 120}, {ID: "consume", Kind: "compute", StartNS: 120, EndNS: 170, DependsOn: []string{"copy", "compute-independent"}}}
	r, err := ProveOperationOverlap(ops, 4)
	if err != nil {
		t.Fatal(err)
	}
	if r.OverlapNanoseconds != 80 || r.CriticalPathNanoseconds != 170 || r.SerialNanoseconds != 250 || r.BytesOverlapped != 4096 {
		t.Fatalf("receipt=%+v", r)
	}
}
func TestProveOperationOverlapRejectsDependencyRace(t *testing.T) {
	_, err := ProveOperationOverlap([]OverlapOperation{{ID: "copy", StartNS: 0, EndNS: 100}, {ID: "consume", StartNS: 99, EndNS: 120, DependsOn: []string{"copy"}}}, 1)
	if err == nil {
		t.Fatal("dependency race accepted")
	}
}
