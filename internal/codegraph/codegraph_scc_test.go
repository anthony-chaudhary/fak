package codegraph

import (
	"reflect"
	"testing"
)

func TestStronglyConnectedComponentsDeterministicAndKindScoped(t *testing.T) {
	g := NewGraph()
	g.AddEdge("b", "a", "calls")
	g.AddEdge("a", "b", "calls")
	g.AddEdge("c", "c", "calls")
	g.AddEdge("d", "a", "references")

	got := g.StronglyConnectedComponents("calls")
	want := [][]NodeID{{"a", "b"}, {"c"}, {"d"}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("components=%v want %v", got, want)
	}
}
