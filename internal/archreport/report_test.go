package archreport

import (
	"bytes"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
)

func TestLateralBlockVertexConnectivityQuantifiesPackageSeparators(t *testing.T) {
	members := []string{"a", "b", "c", "d"}
	cycle := map[string][2]string{"ab": {"a", "b"}, "bc": {"b", "c"}, "cd": {"c", "d"}, "ad": {"a", "d"}, "duplicate": {"b", "a"}}
	for i := 0; i < 50; i++ {
		cut, separator := blockVertexConnectivity(members, cycle)
		if cut != 2 || !reflect.DeepEqual(separator, []string{"a", "c"}) {
			t.Fatalf("cycle run %d cut=%d separator=%v", i, cut, separator)
		}
	}
	clique := map[string][2]string{}
	for i, left := range members {
		for _, right := range members[i+1:] {
			clique[right+left] = [2]string{right, left}
		}
	}
	cut, separator := blockVertexConnectivity(members, clique)
	if cut != 3 || !reflect.DeepEqual(separator, []string{"a", "b", "c"}) {
		t.Fatalf("K4 cut=%d separator=%v", cut, separator)
	}

	asymmetric := map[string][2]string{
		"ab": {"a", "b"}, "ac": {"a", "c"}, "bc": {"b", "c"},
		"bd": {"b", "d"}, "cd": {"c", "d"}, "de": {"d", "e"}, "ce": {"c", "e"},
	}
	asymmetricMembers := []string{"a", "b", "c", "d", "e"}
	cut, separator = blockVertexConnectivity(asymmetricMembers, asymmetric)
	if cut != 2 || !reflect.DeepEqual(separator, []string{"b", "c"}) {
		t.Fatalf("asymmetric cut=%d separator=%v", cut, separator)
	}
	if len(separator) != cut || !separatorDisconnectsOrTrivial(asymmetricMembers, asymmetric, separator) {
		t.Fatalf("separator does not witness cut: cut=%d separator=%v", cut, separator)
	}
}

func separatorDisconnectsOrTrivial(members []string, edges map[string][2]string, separator []string) bool {
	removed := map[string]struct{}{}
	for _, member := range separator {
		removed[member] = struct{}{}
	}
	remaining := make([]string, 0, len(members)-len(separator))
	for _, member := range members {
		if _, gone := removed[member]; !gone {
			remaining = append(remaining, member)
		}
	}
	if len(remaining) <= 1 {
		return true
	}
	seen := map[string]struct{}{remaining[0]: {}}
	queue := []string{remaining[0]}
	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]
		for _, edge := range edges {
			next := ""
			if edge[0] == current {
				next = edge[1]
			} else if edge[1] == current {
				next = edge[0]
			}
			if next == "" {
				continue
			}
			if _, gone := removed[next]; gone {
				continue
			}
			if _, ok := seen[next]; !ok {
				seen[next] = struct{}{}
				queue = append(queue, next)
			}
		}
	}
	return len(seen) != len(remaining)
}

func TestLateralBlockEdgeConnectivityDistinguishesCycleAndClique(t *testing.T) {
	members := []string{"a", "b", "c", "d"}
	cycle := map[string][2]string{"ab": {"a", "b"}, "bc": {"b", "c"}, "cd": {"c", "d"}, "ad": {"a", "d"}}
	cut, pairs, allPairs := blockEdgeConnectivity(members, cycle)
	if cut != 2 || len(pairs) != 6 || len(allPairs) != 6 {
		t.Fatalf("cycle cut=%d pairs=%+v", cut, pairs)
	}
	clique := map[string][2]string{}
	for i := 0; i < len(members); i++ {
		for j := i + 1; j < len(members); j++ {
			clique[members[i]+members[j]] = [2]string{members[i], members[j]}
		}
	}
	cut, pairs, allPairs = blockEdgeConnectivity(members, clique)
	if cut != 3 || len(pairs) != 6 || len(allPairs) != 6 {
		t.Fatalf("clique cut=%d pairs=%+v", cut, pairs)
	}
	clique["duplicate"] = [2]string{"b", "a"}
	cut, _, _ = blockEdgeConnectivity(members, clique)
	if cut != 3 {
		t.Fatalf("duplicate orientation inflated cut=%d", cut)
	}
}

func TestLateralMinimumCutWitnessIsCanonicalAndDeterministic(t *testing.T) {
	members := map[string]struct{}{"a": {}, "b": {}, "c": {}, "d": {}}
	cycle := map[string][2]string{
		"z": {"d", "a"}, "bc": {"b", "c"}, "ab": {"b", "a"}, "cd": {"c", "d"},
		"duplicate-orientation": {"a", "b"},
	}
	want := []LateralCutEdge{{Left: "a", Right: "b"}, {Left: "a", Right: "d"}}
	for i := 0; i < 20; i++ {
		cut, witness, sourceSide, sinkSide := unitEdgeMaxFlow("a", "c", cycle, members)
		if cut != 2 || !reflect.DeepEqual(witness, want) {
			t.Fatalf("run %d: cut=%d witness=%+v want=%+v", i, cut, witness, want)
		}
		if len(witness) != cut {
			t.Fatalf("run %d: witness cardinality=%d cut=%d", i, len(witness), cut)
		}
		if !reflect.DeepEqual(sourceSide, []string{"a"}) || !reflect.DeepEqual(sinkSide, []string{"b", "c", "d"}) {
			t.Fatalf("run %d: source=%v sink=%v", i, sourceSide, sinkSide)
		}
	}

	clique := map[string][2]string{}
	for left := range members {
		for right := range members {
			if left < right {
				clique[left+right] = [2]string{right, left}
			}
		}
	}
	cut, witness, sourceSide, sinkSide := unitEdgeMaxFlow("a", "b", clique, members)
	want = []LateralCutEdge{{Left: "a", Right: "b"}, {Left: "a", Right: "c"}, {Left: "a", Right: "d"}}
	if cut != 3 || !reflect.DeepEqual(witness, want) {
		t.Fatalf("K4 cut=%d witness=%+v want=%+v", cut, witness, want)
	}
	if !reflect.DeepEqual(sourceSide, []string{"a"}) || !reflect.DeepEqual(sinkSide, []string{"b", "c", "d"}) {
		t.Fatalf("K4 source=%v sink=%v", sourceSide, sinkSide)
	}
}

func TestLateralMinimumCutPartitionCoversMembersAndWitnessCrossings(t *testing.T) {
	members := map[string]struct{}{"a": {}, "b": {}, "c": {}, "d": {}, "e": {}}
	edges := map[string][2]string{
		"ab": {"a", "b"}, "ac": {"a", "c"}, "bc": {"b", "c"},
		"bd": {"b", "d"}, "cd": {"c", "d"}, "de": {"d", "e"}, "ce": {"c", "e"},
	}
	for i := 0; i < 20; i++ {
		cut, witness, sourceSide, sinkSide := unitEdgeMaxFlow("a", "e", edges, members)
		if cut != len(witness) || len(sourceSide)+len(sinkSide) != len(members) {
			t.Fatalf("run %d: cut=%d witness=%v source=%v sink=%v", i, cut, witness, sourceSide, sinkSide)
		}
		if !containsString(sourceSide, "a") || containsString(sourceSide, "e") || !containsString(sinkSide, "e") {
			t.Fatalf("run %d: invalid endpoints source=%v sink=%v", i, sourceSide, sinkSide)
		}
		source := map[string]struct{}{}
		for _, member := range sourceSide {
			source[member] = struct{}{}
		}
		for _, edge := range witness {
			_, left := source[edge.Left]
			_, right := source[edge.Right]
			if left == right {
				t.Fatalf("run %d: witness edge %+v does not cross source=%v", i, edge, sourceSide)
			}
		}
	}
}

func containsString(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}

func TestAnalyzeDerivesLateralBiconnectedBlocks(t *testing.T) {
	root := t.TempDir()
	write := func(path, body string) {
		t.Helper()
		p := filepath.Join(root, filepath.FromSlash(path))
		if err := os.MkdirAll(filepath.Dir(p), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0644); err != nil {
			t.Fatal(err)
		}
	}
	write("internal/architest/architest_test.go", `package architest
var tier=map[string]int{"a":2,"b":2,"c":2,"d":2,"e":2,"x":3,"y":3,"z":3}
var tierName=[]string{"zero","primitive","foundation-composite","mechanism"}
`)
	write("internal/a/a.go", `package a
import (_ "github.com/anthony-chaudhary/fak/internal/b"; _ "github.com/anthony-chaudhary/fak/internal/c")
`)
	write("internal/b/b.go", `package b
import _ "github.com/anthony-chaudhary/fak/internal/c"
`)
	write("internal/c/c.go", `package c
import (_ "github.com/anthony-chaudhary/fak/internal/d"; _ "github.com/anthony-chaudhary/fak/internal/e")
`)
	write("internal/d/d.go", `package d
import _ "github.com/anthony-chaudhary/fak/internal/e"
`)
	write("internal/e/e.go", "package e\n")
	write("internal/x/x.go", `package x
import _ "github.com/anthony-chaudhary/fak/internal/y"
`)
	write("internal/y/y.go", `package y
import _ "github.com/anthony-chaudhary/fak/internal/z"
`)
	write("internal/z/z.go", "package z\n")
	r, err := Analyze(root, "")
	if err != nil {
		t.Fatal(err)
	}
	trianglePairs := func(a, b, c string) []LateralCriticalPair {
		return []LateralCriticalPair{
			{Left: a, Right: b, Cut: 2, CutEdges: []LateralCutEdge{{Left: a, Right: b}, {Left: a, Right: c}}, SourceSide: []string{a}, SinkSide: []string{b, c}},
			{Left: a, Right: c, Cut: 2, CutEdges: []LateralCutEdge{{Left: a, Right: b}, {Left: a, Right: c}}, SourceSide: []string{a}, SinkSide: []string{b, c}},
			{Left: b, Right: c, Cut: 2, CutEdges: []LateralCutEdge{{Left: a, Right: b}, {Left: b, Right: c}}, SourceSide: []string{b}, SinkSide: []string{a, c}},
		}
	}
	abc, cde := trianglePairs("a", "b", "c"), trianglePairs("c", "d", "e")
	want := []LateralBiconnectedBlock{
		{Tier: 2, TierName: "foundation-composite", Members: []string{"a", "b", "c"}, MemberCount: 3, EdgeCount: 3, MinEdgeCut: 2, MinVertexCut: 2, CriticalSeparator: []string{"a", "b"}, CriticalPairs: abc, PairCuts: abc},
		{Tier: 2, TierName: "foundation-composite", Members: []string{"c", "d", "e"}, MemberCount: 3, EdgeCount: 3, MinEdgeCut: 2, MinVertexCut: 2, CriticalSeparator: []string{"c", "d"}, CriticalPairs: cde, PairCuts: cde},
	}
	if !reflect.DeepEqual(r.LateralBiconnectedBlocks, want) {
		t.Fatalf("blocks=%+v want=%+v", r.LateralBiconnectedBlocks, want)
	}
	scoped, err := Analyze(root, "c")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(scoped.LateralBiconnectedBlocks, want) {
		t.Fatalf("scoped=%+v", scoped.LateralBiconnectedBlocks)
	}
	chain, err := Analyze(root, "x")
	if err != nil {
		t.Fatal(err)
	}
	if len(chain.LateralBiconnectedBlocks) != 0 {
		t.Fatalf("chain=%+v", chain.LateralBiconnectedBlocks)
	}
}

func TestAnalyzeIdentifiesLateralArticulationPoints(t *testing.T) {
	root := t.TempDir()
	write := func(path, body string) {
		t.Helper()
		p := filepath.Join(root, filepath.FromSlash(path))
		if err := os.MkdirAll(filepath.Dir(p), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0644); err != nil {
			t.Fatal(err)
		}
	}
	write("internal/architest/architest_test.go", `package architest
var tier=map[string]int{"a":2,"b":2,"c":2,"d":2,"center":3,"p":3,"q":3,"r":3,"x":4,"y":4,"z":4}
var tierName=[]string{"zero","primitive","foundation-composite","mechanism","composer"}
`)
	write("internal/a/a.go", `package a
import _ "github.com/anthony-chaudhary/fak/internal/b"
`)
	write("internal/b/b.go", `package b
import _ "github.com/anthony-chaudhary/fak/internal/c"
`)
	write("internal/c/c.go", `package c
import _ "github.com/anthony-chaudhary/fak/internal/d"
`)
	write("internal/d/d.go", "package d\n")
	write("internal/center/center.go", `package center
import (_ "github.com/anthony-chaudhary/fak/internal/p"; _ "github.com/anthony-chaudhary/fak/internal/q"; _ "github.com/anthony-chaudhary/fak/internal/r")
`)
	for _, leaf := range []string{"p", "q", "r"} {
		write("internal/"+leaf+"/"+leaf+".go", "package "+leaf+"\n")
	}
	write("internal/x/x.go", `package x
import (_ "github.com/anthony-chaudhary/fak/internal/y"; _ "github.com/anthony-chaudhary/fak/internal/z")
`)
	write("internal/y/y.go", `package y
import _ "github.com/anthony-chaudhary/fak/internal/z"
`)
	write("internal/z/z.go", "package z\n")
	r, err := Analyze(root, "")
	if err != nil {
		t.Fatal(err)
	}
	want := []LateralArticulationPoint{
		{Tier: 3, TierName: "mechanism", Name: "center", Fragments: [][]string{{"p"}, {"q"}, {"r"}}, FragmentCount: 3, CouplingPairs: 3},
		{Tier: 2, TierName: "foundation-composite", Name: "b", Fragments: [][]string{{"a"}, {"c", "d"}}, FragmentCount: 2, CouplingPairs: 2},
		{Tier: 2, TierName: "foundation-composite", Name: "c", Fragments: [][]string{{"a", "b"}, {"d"}}, FragmentCount: 2, CouplingPairs: 2},
	}
	if !reflect.DeepEqual(r.LateralArticulationPoints, want) {
		t.Fatalf("points=%+v want=%+v", r.LateralArticulationPoints, want)
	}
	scoped, err := Analyze(root, "a")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(scoped.LateralArticulationPoints, want[1:]) {
		t.Fatalf("scoped=%+v", scoped.LateralArticulationPoints)
	}
	cycle, err := Analyze(root, "x")
	if err != nil {
		t.Fatal(err)
	}
	if len(cycle.LateralArticulationPoints) != 0 {
		t.Fatalf("cycle points=%+v", cycle.LateralArticulationPoints)
	}
}

func TestAnalyzeIdentifiesLateralBridgesAndInducedCouplings(t *testing.T) {
	root := t.TempDir()
	write := func(path, body string) {
		t.Helper()
		p := filepath.Join(root, filepath.FromSlash(path))
		if err := os.MkdirAll(filepath.Dir(p), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0644); err != nil {
			t.Fatal(err)
		}
	}
	write("internal/architest/architest_test.go", `package architest
var tier=map[string]int{"a":2,"b":2,"c":2,"d":2,"x":3,"y":3,"z":3}
var tierName=[]string{"zero","primitive","foundation-composite","mechanism"}
`)
	write("internal/a/a.go", `package a
import _ "github.com/anthony-chaudhary/fak/internal/b"
`)
	write("internal/b/b.go", `package b
import _ "github.com/anthony-chaudhary/fak/internal/c"
`)
	write("internal/c/c.go", `package c
import _ "github.com/anthony-chaudhary/fak/internal/d"
`)
	write("internal/d/d.go", "package d\n")
	write("internal/x/x.go", `package x
import (_ "github.com/anthony-chaudhary/fak/internal/y"; _ "github.com/anthony-chaudhary/fak/internal/z")
`)
	write("internal/y/y.go", `package y
import _ "github.com/anthony-chaudhary/fak/internal/z"
`)
	write("internal/z/z.go", "package z\n")
	r, err := Analyze(root, "")
	if err != nil {
		t.Fatal(err)
	}
	want := []LateralBridge{
		{Tier: 2, TierName: "foundation-composite", From: "b", To: "c", Left: "b", Right: "c", LeftSide: []string{"a", "b"}, RightSide: []string{"c", "d"}, CouplingPairs: 4},
		{Tier: 2, TierName: "foundation-composite", From: "a", To: "b", Left: "a", Right: "b", LeftSide: []string{"a"}, RightSide: []string{"b", "c", "d"}, CouplingPairs: 3},
		{Tier: 2, TierName: "foundation-composite", From: "c", To: "d", Left: "c", Right: "d", LeftSide: []string{"a", "b", "c"}, RightSide: []string{"d"}, CouplingPairs: 3},
	}
	if !reflect.DeepEqual(r.LateralBridges, want) {
		t.Fatalf("bridges=%+v want=%+v", r.LateralBridges, want)
	}
	scoped, err := Analyze(root, "b")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(scoped.LateralBridges, want) {
		t.Fatalf("scoped=%+v", scoped.LateralBridges)
	}
	diamond, err := Analyze(root, "x")
	if err != nil {
		t.Fatal(err)
	}
	if len(diamond.LateralBridges) != 0 {
		t.Fatalf("diamond bridges=%+v", diamond.LateralBridges)
	}
}

func TestAnalyzeDerivesDeterministicLateralComponents(t *testing.T) {
	root := t.TempDir()
	write := func(path, body string) {
		t.Helper()
		p := filepath.Join(root, filepath.FromSlash(path))
		if err := os.MkdirAll(filepath.Dir(p), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0644); err != nil {
			t.Fatal(err)
		}
	}
	write("internal/architest/architest_test.go", `package architest
var tier=map[string]int{"a":2,"b":2,"c":2,"d":2,"x":3,"y":3,"alone":2}
var tierName=[]string{"zero","primitive","foundation-composite","mechanism"}
`)
	write("internal/a/a.go", `package a
import (_ "github.com/anthony-chaudhary/fak/internal/b"; _ "github.com/anthony-chaudhary/fak/internal/c")
`)
	write("internal/b/b.go", `package b
import _ "github.com/anthony-chaudhary/fak/internal/d"
`)
	write("internal/c/c.go", `package c
import _ "github.com/anthony-chaudhary/fak/internal/d"
`)
	write("internal/d/d.go", "package d\n")
	write("internal/x/x.go", `package x
import _ "github.com/anthony-chaudhary/fak/internal/y"
`)
	write("internal/y/y.go", "package y\n")
	write("internal/alone/alone.go", "package alone\n")
	r, err := Analyze(root, "")
	if err != nil {
		t.Fatal(err)
	}
	want := []LateralComponent{
		{Tier: 2, TierName: "foundation-composite", Members: []string{"a", "b", "c", "d"}, MemberCount: 4, EdgeCount: 4},
		{Tier: 3, TierName: "mechanism", Members: []string{"x", "y"}, MemberCount: 2, EdgeCount: 1},
	}
	if !reflect.DeepEqual(r.LateralComponents, want) {
		t.Fatalf("components=%+v want=%+v", r.LateralComponents, want)
	}
	scoped, err := Analyze(root, "b")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(scoped.LateralComponents, want[:1]) {
		t.Fatalf("scoped=%+v", scoped.LateralComponents)
	}
	alone, err := Analyze(root, "alone")
	if err != nil {
		t.Fatal(err)
	}
	if len(alone.LateralComponents) != 0 {
		t.Fatalf("alone=%+v", alone.LateralComponents)
	}
}

func TestAnalyzeTypesEveryLiveArchitectureEdgeDirection(t *testing.T) {
	root := t.TempDir()
	write := func(path, body string) {
		t.Helper()
		p := filepath.Join(root, filepath.FromSlash(path))
		if err := os.MkdirAll(filepath.Dir(p), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0644); err != nil {
			t.Fatal(err)
		}
	}
	write("internal/architest/architest_test.go", `package architest
var tier=map[string]int{"root":1,"low":2,"peer":2,"high":3,"stale":3}
var tierName=[]string{"zero","primitive","foundation-composite","mechanism"}
`)
	write("internal/root/root.go", "package root\n")
	write("internal/peer/peer.go", "package peer\n")
	write("internal/high/high.go", "package high\n")
	write("internal/low/low.go", `package low
import (
 _ "github.com/anthony-chaudhary/fak/internal/root"
 _ "github.com/anthony-chaudhary/fak/internal/peer"
 _ "github.com/anthony-chaudhary/fak/internal/high"
 _ "github.com/anthony-chaudhary/fak/internal/stale"
)
`)
	r, err := Analyze(root, "low")
	if err != nil {
		t.Fatal(err)
	}
	want := []ArchitectureEdge{
		{From: "low", FromTier: 2, FromTierName: "foundation-composite", To: "high", ToTier: 3, ToTierName: "mechanism", TierDelta: 1, Direction: "upward"},
		{From: "low", FromTier: 2, FromTierName: "foundation-composite", To: "peer", ToTier: 2, ToTierName: "foundation-composite", TierDelta: 0, Direction: "lateral"},
		{From: "low", FromTier: 2, FromTierName: "foundation-composite", To: "root", ToTier: 1, ToTierName: "primitive", TierDelta: -1, Direction: "rootward"},
	}
	if !reflect.DeepEqual(r.Edges, want) {
		t.Fatalf("edges=%+v want=%+v", r.Edges, want)
	}
	if len(r.Leaves) != 1 || len(r.Leaves[0].ViolationEdges) != 1 || r.Leaves[0].ViolationEdges[0].To != "high" {
		t.Fatalf("leaf=%+v", r.Leaves)
	}
	upward := []string{}
	for _, edge := range r.Edges {
		if edge.Direction == "upward" {
			upward = append(upward, edge.From+" -> "+edge.To)
		}
	}
	if !reflect.DeepEqual(upward, []string{"low -> high"}) {
		t.Fatalf("upward=%v", upward)
	}
}

func TestAnalyzeReportsUpwardEdgeAndLegalReverse(t *testing.T) {
	root := t.TempDir()
	write := func(path, body string) {
		t.Helper()
		p := filepath.Join(root, filepath.FromSlash(path))
		if err := os.MkdirAll(filepath.Dir(p), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0644); err != nil {
			t.Fatal(err)
		}
	}
	write("internal/architest/architest_test.go", `package architest
var tier=map[string]int{"abi":0,"primitive":1,"composite":2}
var tierName=[]string{"root","primitive","foundation-composite"}
`)
	write("internal/abi/abi.go", "package abi\n")
	write("internal/primitive/primitive.go", `package primitive
import _ "github.com/anthony-chaudhary/fak/internal/composite"
`)
	write("internal/composite/composite.go", `package composite
import _ "github.com/anthony-chaudhary/fak/internal/primitive"
`)
	r, err := Analyze(root, "")
	if err != nil {
		t.Fatal(err)
	}
	if r.Violations != 1 {
		t.Fatalf("violations=%d report=%+v", r.Violations, r)
	}
	if len(r.Leaves) != 3 {
		t.Fatalf("leaves=%d", len(r.Leaves))
	}
	var p, c Leaf
	for _, l := range r.Leaves {
		if l.Name == "primitive" {
			p = l
		}
		if l.Name == "composite" {
			c = l
		}
	}
	wantEdge := ViolationEdge{From: "primitive", FromTier: 1, FromTierName: "primitive", To: "composite", ToTier: 2, ToTierName: "foundation-composite", TierDistance: 1}
	if len(p.ViolationEdges) != 1 || p.ViolationEdges[0] != wantEdge || !reflect.DeepEqual(p.Violations, []string{"primitive -> composite"}) || p.ImportFloor != 2 {
		t.Fatalf("primitive=%+v", p)
	}
	if len(c.Violations) != 0 || c.ImportFloor != 1 {
		t.Fatalf("composite=%+v", c)
	}
}

func TestAnalyzeScopesLeaf(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "internal", "architest"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "internal", "architest", "architest_test.go"), []byte(`package architest
var tier=map[string]int{"architest":2}
var tierName=[]string{"root","primitive","foundation-composite"}
`), 0644); err != nil {
		t.Fatal(err)
	}
	r, err := Analyze(root, "architest")
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Leaves) != 1 || r.Leaves[0].Name != "architest" || r.Leaves[0].ImportFloor != 1 {
		t.Fatalf("%+v", r)
	}
}

func TestAnalyzeDerivesDirectDependentsAndRanksHotspots(t *testing.T) {
	root := t.TempDir()
	write := func(path, body string) {
		t.Helper()
		p := filepath.Join(root, filepath.FromSlash(path))
		if err := os.MkdirAll(filepath.Dir(p), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0644); err != nil {
			t.Fatal(err)
		}
	}
	write("internal/architest/architest_test.go", `package architest
var tier=map[string]int{"abi":0,"alpha":2,"beta":2,"gamma":2,"delta":2}
var tierName=[]string{"root","primitive","foundation-composite"}
`)
	write("internal/abi/abi.go", "package abi\n")
	write("internal/alpha/alpha.go", `package alpha
import (
 _ "github.com/anthony-chaudhary/fak/internal/abi"
 _ "github.com/anthony-chaudhary/fak/internal/beta"
)
`)
	write("internal/beta/beta.go", `package beta
import _ "github.com/anthony-chaudhary/fak/internal/abi"
`)
	write("internal/gamma/gamma.go", `package gamma
import _ "github.com/anthony-chaudhary/fak/internal/beta"
`)
	write("internal/delta/delta.go", "package delta\n")

	r, err := Analyze(root, "")
	if err != nil {
		t.Fatal(err)
	}
	wantHotspots := []Hotspot{{Name: "abi", FanIn: 2}, {Name: "beta", FanIn: 2}}
	if !reflect.DeepEqual(r.Hotspots, wantHotspots) {
		t.Fatalf("hotspots=%+v want=%+v", r.Hotspots, wantHotspots)
	}
	wantBlastHotspots := []BlastHotspot{{Name: "abi", BlastRadius: 3, MaxHops: 2}, {Name: "beta", BlastRadius: 2, MaxHops: 1}}
	if !reflect.DeepEqual(r.BlastHotspots, wantBlastHotspots) {
		t.Fatalf("blast hotspots=%+v want=%+v", r.BlastHotspots, wantBlastHotspots)
	}
	byName := map[string]Leaf{}
	for _, leaf := range r.Leaves {
		byName[leaf.Name] = leaf
	}
	if got, want := byName["abi"].Dependents, []string{"alpha", "beta"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("abi dependents=%v want=%v", got, want)
	}
	if got, want := byName["beta"].Dependents, []string{"alpha", "gamma"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("beta dependents=%v want=%v", got, want)
	}
	if len(byName["delta"].Dependents) != 0 {
		t.Fatalf("delta dependents=%v", byName["delta"].Dependents)
	}
	if got, want := byName["abi"].TransitiveDependents, []string{"alpha", "beta", "gamma"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("abi transitive dependents=%v want=%v", got, want)
	}
	wantPaths := []BlastPath{
		{Dependent: "alpha", Path: []string{"abi", "alpha"}},
		{Dependent: "beta", Path: []string{"abi", "beta"}},
		{Dependent: "gamma", Path: []string{"abi", "beta", "gamma"}},
	}
	if got := byName["abi"].BlastPaths; !reflect.DeepEqual(got, wantPaths) {
		t.Fatalf("abi blast paths=%v want=%v", got, wantPaths)
	}
	if got := byName["abi"].BlastRadius; got != 3 {
		t.Fatalf("abi blast radius=%d want=3", got)
	}
	if got, want := byName["beta"].TransitiveDependents, []string{"alpha", "gamma"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("beta transitive dependents=%v want=%v", got, want)
	}
	if got := byName["delta"].TransitiveDependents; got == nil || len(got) != 0 || byName["delta"].BlastRadius != 0 || byName["delta"].BlastPaths == nil || len(byName["delta"].BlastPaths) != 0 {
		t.Fatalf("delta transitive dependents=%v blast radius=%d blast paths=%v", got, byName["delta"].BlastRadius, byName["delta"].BlastPaths)
	}

	scoped, err := Analyze(root, "abi")
	if err != nil {
		t.Fatal(err)
	}
	if len(scoped.Leaves) != 1 || !reflect.DeepEqual(scoped.Leaves[0].Dependents, []string{"alpha", "beta"}) || !reflect.DeepEqual(scoped.Leaves[0].TransitiveDependents, []string{"alpha", "beta", "gamma"}) || scoped.Leaves[0].BlastRadius != 3 {
		t.Fatalf("scoped=%+v", scoped)
	}
	if len(scoped.Hotspots) != 0 || len(scoped.BlastHotspots) != 0 {
		t.Fatalf("scoped hotspots=%+v blast hotspots=%+v", scoped.Hotspots, scoped.BlastHotspots)
	}
	if scoped.Violations != 0 {
		t.Fatalf("scoped violations=%d", scoped.Violations)
	}
}

func TestAnalyzeAdversarialInputs(t *testing.T) {
	tests := []struct {
		name       string
		contract   string
		leaf       string
		files      map[string]string
		wantErr    string
		wantDeps   []string
		wantFloor  int
		violations int
	}{
		{
			name:     "malformed contract",
			contract: "package architest\nvar tier = map[string]int{",
			wantErr:  "parse architecture contract",
		},
		{
			name:     "missing tier table",
			contract: "package architest\nvar tierName=[]string{\"root\"}\n",
			wantErr:  "missing tier or tierName",
		},
		{
			name:     "unknown scoped leaf",
			contract: "package architest\nvar tier=map[string]int{\"known\":1}\nvar tierName=[]string{\"root\",\"primitive\"}\n",
			leaf:     "hostile-not-declared",
			wantErr:  `leaf "hostile-not-declared" has no tier declaration`,
		},
		{
			name:     "malformed leaf source",
			contract: "package architest\nvar tier=map[string]int{\"broken\":1}\nvar tierName=[]string{\"root\",\"primitive\"}\n",
			files:    map[string]string{"internal/broken/broken.go": "package broken\nimport ("},
			wantErr:  "broken.go",
		},
		{
			name:     "deduplicates and collapses nested imports",
			contract: "package architest\nvar tier=map[string]int{\"source\":1,\"target\":2}\nvar tierName=[]string{\"root\",\"primitive\",\"foundation-composite\"}\n",
			leaf:     "source",
			files: map[string]string{
				"internal/source/a.go":      "package source\nimport _ \"github.com/anthony-chaudhary/fak/internal/target/subpackage\"\n",
				"internal/source/b.go":      "package source\nimport _ \"github.com/anthony-chaudhary/fak/internal/target\"\n",
				"internal/source/a_test.go": "package source\nimport _ \"github.com/anthony-chaudhary/fak/internal/ignored\"\n",
				"internal/target/target.go": "package target\n",
			},
			wantDeps:   []string{"target"},
			wantFloor:  2,
			violations: 1,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			root := t.TempDir()
			writeArchitectureFixture(t, root, "internal/architest/architest_test.go", tt.contract)
			for path, body := range tt.files {
				writeArchitectureFixture(t, root, path, body)
			}
			r, err := Analyze(root, tt.leaf)
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(filepath.ToSlash(err.Error()), tt.wantErr) {
					t.Fatalf("err=%v want substring %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if len(r.Leaves) != 1 {
				t.Fatalf("leaves=%d report=%+v", len(r.Leaves), r)
			}
			if !reflect.DeepEqual(r.Leaves[0].Dependencies, tt.wantDeps) {
				t.Fatalf("dependencies=%v want=%v", r.Leaves[0].Dependencies, tt.wantDeps)
			}
			if r.Leaves[0].ImportFloor != tt.wantFloor || r.Violations != tt.violations {
				t.Fatalf("floor=%d violations=%d", r.Leaves[0].ImportFloor, r.Violations)
			}
		})
	}
}

func TestAnalyzeErrorsNameRecovery(t *testing.T) {
	tests := []struct {
		name     string
		contract string
		leaf     string
		files    map[string]string
		want     []string
	}{
		{
			name:     "contract syntax",
			contract: "package architest\nvar tier = map[string]int{",
			want:     []string{"parse architecture contract", "repair the Go syntax before reporting"},
		},
		{
			name:     "contract declaration",
			contract: "package architest\nvar tierName=[]string{\"root\"}\n",
			want:     []string{"missing tier or tierName", "restore both declarations"},
		},
		{
			name:     "unknown leaf",
			contract: "package architest\nvar tier=map[string]int{\"known\":1}\nvar tierName=[]string{\"root\",\"primitive\"}\n",
			leaf:     "unknown",
			want:     []string{"has no tier declaration", "choose a declared leaf", "or add its tier there"},
		},
		{
			name:     "leaf syntax",
			contract: "package architest\nvar tier=map[string]int{\"broken\":1}\nvar tierName=[]string{\"root\",\"primitive\"}\n",
			files:    map[string]string{"internal/broken/broken.go": "package broken\nimport ("},
			want:     []string{"parse imports", "broken.go", "repair the Go syntax before reporting"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			root := t.TempDir()
			writeArchitectureFixture(t, root, "internal/architest/architest_test.go", tt.contract)
			for path, body := range tt.files {
				writeArchitectureFixture(t, root, path, body)
			}
			_, err := Analyze(root, tt.leaf)
			if err == nil {
				t.Fatal("expected error")
			}
			for _, want := range tt.want {
				if !strings.Contains(err.Error(), want) {
					t.Fatalf("error %q does not name recovery %q", err, want)
				}
			}
		})
	}
}

func TestAnalyzeRanksSinkCandidatesByTierGapThenName(t *testing.T) {
	root := t.TempDir()
	writeArchitectureFixture(t, root, "internal/architest/architest_test.go", `package architest
var tier=map[string]int{"abi":0,"alpha":4,"beta":3,"gamma":3,"delta":2}
var tierName=[]string{"root","primitive","foundation-composite","mechanism","composer"}
`)
	for _, leaf := range []string{"abi", "alpha", "beta", "gamma", "delta"} {
		writeArchitectureFixture(t, root, "internal/"+leaf+"/leaf.go", "package "+leaf+"\n")
	}
	report, err := Analyze(root, "")
	if err != nil {
		t.Fatal(err)
	}
	want := []SinkCandidate{
		{Name: "alpha", DeclaredTier: 4, DeclaredTierName: "composer", ImportFloor: 1, ImportFloorName: "primitive", TierGap: 3},
		{Name: "beta", DeclaredTier: 3, DeclaredTierName: "mechanism", ImportFloor: 1, ImportFloorName: "primitive", TierGap: 2},
		{Name: "gamma", DeclaredTier: 3, DeclaredTierName: "mechanism", ImportFloor: 1, ImportFloorName: "primitive", TierGap: 2},
	}
	if !reflect.DeepEqual(report.SinkCandidates, want) {
		t.Fatalf("candidates=%+v want=%+v", report.SinkCandidates, want)
	}
	for _, leaf := range report.Leaves {
		if leaf.TierGap != leaf.DeclaredTier-leaf.ImportFloor {
			t.Fatalf("leaf=%+v", leaf)
		}
	}
	scoped, err := Analyze(root, "alpha")
	if err != nil {
		t.Fatal(err)
	}
	if len(scoped.SinkCandidates) != 0 || len(scoped.Leaves) != 1 || scoped.Leaves[0].TierGap != 3 {
		t.Fatalf("scoped=%+v", scoped)
	}
}

func TestAnalyzeStaleDeclarationDoesNotSuppressHealthyLeaves(t *testing.T) {
	root := t.TempDir()
	writeArchitectureFixture(t, root, "internal/architest/architest_test.go", `package architest
var tier=map[string]int{"healthy":1,"stale":2}
var tierName=[]string{"root","primitive","foundation-composite"}
`)
	writeArchitectureFixture(t, root, "internal/healthy/healthy.go", "package healthy\n")

	full, err := Analyze(root, "")
	if err != nil {
		t.Fatal(err)
	}
	if len(full.Leaves) != 1 || full.Leaves[0].Name != "healthy" {
		t.Fatalf("healthy leaves suppressed: %+v", full.Leaves)
	}
	wantDiagnostic := Diagnostic{
		Kind:     DiagnosticStaleTierDeclaration,
		Leaf:     "stale",
		Message:  "declared package directory " + filepath.Join(root, "internal", "stale") + " does not exist",
		Recovery: "create the package or remove its stale tier declaration",
	}
	if !reflect.DeepEqual(full.Diagnostics, []Diagnostic{wantDiagnostic}) {
		t.Fatalf("diagnostics=%+v want=%+v", full.Diagnostics, wantDiagnostic)
	}

	healthy, err := Analyze(root, "healthy")
	if err != nil {
		t.Fatal(err)
	}
	if len(healthy.Leaves) != 1 || healthy.Leaves[0].Name != "healthy" || len(healthy.Diagnostics) != 0 {
		t.Fatalf("healthy scoped report=%+v", healthy)
	}

	stale, err := Analyze(root, "stale")
	if err != nil {
		t.Fatal(err)
	}
	if len(stale.Leaves) != 0 || !reflect.DeepEqual(stale.Diagnostics, []Diagnostic{wantDiagnostic}) {
		t.Fatalf("stale scoped report=%+v", stale)
	}
}

func TestAnalyzeDeterminismUnderConcurrency(t *testing.T) {
	root := t.TempDir()
	writeArchitectureFixture(t, root, "internal/architest/architest_test.go", `package architest
var tier=map[string]int{"abi":0,"alpha":2,"beta":2,"gamma":2}
var tierName=[]string{"root","primitive","foundation-composite"}
`)
	writeArchitectureFixture(t, root, "internal/abi/abi.go", "package abi\n")
	writeArchitectureFixture(t, root, "internal/alpha/alpha.go", `package alpha
import (
 _ "github.com/anthony-chaudhary/fak/internal/beta"
 _ "github.com/anthony-chaudhary/fak/internal/abi"
)
`)
	writeArchitectureFixture(t, root, "internal/beta/beta.go", `package beta
import _ "github.com/anthony-chaudhary/fak/internal/abi"
`)
	writeArchitectureFixture(t, root, "internal/gamma/gamma.go", `package gamma
import _ "github.com/anthony-chaudhary/fak/internal/beta"
`)

	wantReport, err := Analyze(root, "")
	if err != nil {
		t.Fatal(err)
	}
	want, err := wantReport.JSON()
	if err != nil {
		t.Fatal(err)
	}

	const workers = 16
	results := make(chan []byte, workers)
	errs := make(chan error, workers)
	var wg sync.WaitGroup
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			report, err := Analyze(root, "")
			if err != nil {
				errs <- err
				return
			}
			raw, err := report.JSON()
			if err != nil {
				errs <- err
				return
			}
			results <- raw
		}()
	}
	wg.Wait()
	close(results)
	close(errs)
	for err := range errs {
		t.Fatal(err)
	}
	seen := 0
	for got := range results {
		seen++
		if !bytes.Equal(got, want) {
			t.Fatalf("non-deterministic report\nwant:\n%s\ngot:\n%s", want, got)
		}
	}
	if seen != workers {
		t.Fatalf("results=%d want=%d", seen, workers)
	}
}

func writeArchitectureFixture(t *testing.T, root, path, body string) {
	t.Helper()
	p := filepath.Join(root, filepath.FromSlash(path))
	if err := os.MkdirAll(filepath.Dir(p), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(body), 0644); err != nil {
		t.Fatal(err)
	}
}

func TestAnalyzeSortsTypedViolationEdgesAndKeepsStringCompatibility(t *testing.T) {
	root := t.TempDir()
	writeArchitectureFixture(t, root, "internal/architest/architest_test.go", `package architest
var tier=map[string]int{"alpha":1,"zeta":3,"beta":2}
var tierName=[]string{"root","primitive","foundation-composite","mechanism"}
`)
	writeArchitectureFixture(t, root, "internal/alpha/alpha.go", `package alpha
import (
 _ "github.com/anthony-chaudhary/fak/internal/zeta"
 _ "github.com/anthony-chaudhary/fak/internal/beta"
)
`)
	writeArchitectureFixture(t, root, "internal/zeta/zeta.go", "package zeta\n")
	writeArchitectureFixture(t, root, "internal/beta/beta.go", "package beta\n")
	r, err := Analyze(root, "alpha")
	if err != nil {
		t.Fatal(err)
	}
	want := []ViolationEdge{
		{From: "alpha", FromTier: 1, FromTierName: "primitive", To: "zeta", ToTier: 3, ToTierName: "mechanism", TierDistance: 2},
		{From: "alpha", FromTier: 1, FromTierName: "primitive", To: "beta", ToTier: 2, ToTierName: "foundation-composite", TierDistance: 1},
	}
	if len(r.Leaves) != 1 || !reflect.DeepEqual(r.Leaves[0].ViolationEdges, want) || !reflect.DeepEqual(r.Leaves[0].Violations, []string{"alpha -> zeta", "alpha -> beta"}) || r.MaxViolationDistance != 2 {
		t.Fatalf("report=%+v", r)
	}
	raw, err := r.JSON()
	if err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{`"violation_edges"`, `"from_tier_name"`, `"violations"`, `"tier_distance"`, `"max_violation_distance"`} {
		if !bytes.Contains(raw, []byte(key)) {
			t.Fatalf("JSON missing %s: %s", key, raw)
		}
	}
}
