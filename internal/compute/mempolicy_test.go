package compute

import (
	"reflect"
	"testing"
)

// TestParseNumaMapsPolicy pins the numa_maps policy-token contract: only MPOL_BIND
// carries a node set (it is the only strict policy — the one that OOM-kills instead
// of spilling); prefer/interleave/default are reported by label alone.
func TestParseNumaMapsPolicy(t *testing.T) {
	cases := []struct {
		line  string
		label string
		nodes []int
	}{
		{"7f5851000000 bind:0 anon=14 dirty=14 N0=14 kernelpagesize_kB=4", "bind:0", []int{0}},
		{"7f5851000000 bind:0-1,4 file=/x", "bind:0-1,4", []int{0, 1, 4}},
		{"55fa10c00000 default file=/usr/bin/fak mapped=203", "default", nil},
		{"7f5851000000 interleave:0-3 anon=2", "interleave:0-3", nil},
		{"7f5851000000 prefer:1 anon=2", "prefer:1", nil},
		{"garbage", "", nil},
		{"", "", nil},
	}
	for _, c := range cases {
		label, nodes := parseNumaMapsPolicy(c.line)
		if label != c.label || !reflect.DeepEqual(nodes, c.nodes) {
			t.Errorf("parseNumaMapsPolicy(%q) = %q %v, want %q %v", c.line, label, nodes, c.label, c.nodes)
		}
	}
}

// TestParseNodeList covers the kernel node-list grammar and its rejects.
func TestParseNodeList(t *testing.T) {
	cases := []struct {
		in   string
		want []int
	}{
		{"0", []int{0}},
		{"0-3", []int{0, 1, 2, 3}},
		{"0-1,4,6-7", []int{0, 1, 4, 6, 7}},
		{" 0-1 ", []int{0, 1}},
		{"", nil},
		{"x", nil},
		{"3-1", nil},
		{"-1", nil},
	}
	for _, c := range cases {
		if got := parseNodeList(c.in); !reflect.DeepEqual(got, c.want) {
			t.Errorf("parseNodeList(%q) = %v, want %v", c.in, got, c.want)
		}
	}
}

// TestFormatNodeList checks the compact round-trip rendering used in log lines.
func TestFormatNodeList(t *testing.T) {
	cases := []struct {
		in   []int
		want string
	}{
		{[]int{0}, "0"},
		{[]int{0, 1, 2, 3}, "0-3"},
		{[]int{0, 1, 4, 6, 7}, "0-1,4,6-7"},
		{nil, ""},
	}
	for _, c := range cases {
		if got := formatNodeList(c.in); got != c.want {
			t.Errorf("formatNodeList(%v) = %q, want %q", c.in, got, c.want)
		}
	}
}

// TestSubsetOf pins the sorted-subset helper the confinement decision rests on.
func TestSubsetOf(t *testing.T) {
	if !subsetOf([]int{0, 2}, []int{0, 1, 2, 3}) {
		t.Error("subsetOf({0,2},{0..3}) should be true")
	}
	if subsetOf([]int{0, 4}, []int{0, 1, 2, 3}) {
		t.Error("subsetOf({0,4},{0..3}) should be false")
	}
	if !subsetOf(nil, []int{1}) {
		t.Error("empty set is a subset of anything")
	}
}

// TestParseNodeMeminfoFree parses the sysfs per-node meminfo shape.
func TestParseNodeMeminfoFree(t *testing.T) {
	doc := "Node 0 MemTotal:       131651468 kB\nNode 0 MemFree:        59976508 kB\nNode 0 MemUsed:        71674960 kB\n"
	got, ok := parseNodeMeminfoFree(doc)
	if !ok || got != 59976508*1024 {
		t.Errorf("parseNodeMeminfoFree = %d,%v want %d,true", got, ok, int64(59976508*1024))
	}
	if _, ok := parseNodeMeminfoFree("Node 0 MemTotal: 1 kB\n"); ok {
		t.Error("missing MemFree should report !ok")
	}
}

// TestReadHostMemStatusSmoke exercises the real probe end-to-end: it must never
// panic, and a machine that is NOT numa-confined must not claim confinement.
func TestReadHostMemStatusSmoke(t *testing.T) {
	st := ReadHostMemStatus()
	if st.Constrained && st.PolicyNodes == "" {
		t.Error("constrained status must name its nodes")
	}
}
