package enumlint

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeFixture(t *testing.T, src string) string {
	t.Helper()
	root := t.TempDir()
	dir := filepath.Join(root, "internal", "fixture")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "fixture.go"), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	return root
}

func TestDiscoveryAndRulesFixtures(t *testing.T) {
	const src = `package fixture

type State string
const (
 StateNew State = "new"
 StateRunning State = "running"
 StateDone State = "done"
)
type Ordinal int
const (
 OrdinalZero Ordinal = iota
 OrdinalOne
 OrdinalTwo
)
type Open struct{ X int }
const StructValue = Open{}

func switchHole(s State) string {
 switch s {
 case StateNew: return "new"
 case StateRunning: return "running"
 }
 return ""
}
func switchDefault(s State) string {
 switch s { case StateNew: return "new"; default: return "other" }
}
var mapHole = map[State]string{StateNew: "new", StateRunning: "running"}
var arrayHole = []Ordinal{OrdinalZero, OrdinalOne}
`
	root := writeFixture(t, src)
	pkgs, unparsed, err := Discover(root, Config{})
	if err != nil {
		t.Fatal(err)
	}
	if len(unparsed) != 0 || len(pkgs) != 1 {
		t.Fatalf("packages=%d unparsed=%v", len(pkgs), unparsed)
	}
	got := map[string]*Enum{}
	for _, e := range pkgs[0].Enums {
		got[e.Name] = e
	}
	if got["Open"] != nil {
		t.Fatal("struct-backed type was incorrectly discovered as a closed enum")
	}
	state := got["State"]
	if state == nil || len(state.Members) != 3 {
		t.Fatalf("State=%#v", state)
	}
	ordinal := got["Ordinal"]
	if ordinal == nil || len(ordinal.Members) != 3 || ordinal.Members[0].Val != "" || ordinal.Members[1].Val != "" || ordinal.Members[2].Val != "" {
		t.Fatalf("iota members=%#v", ordinal)
	}
	rep, err := Scan(root, Config{LiteralMinMembers: 2, LiteralMaxOmitted: 2})
	if err != nil {
		t.Fatal(err)
	}
	var sw, mp, arr bool
	for _, f := range rep.Findings {
		if f.Owner == "switchHole" && f.Rule == RuleSwitch && len(f.Missing) == 1 && f.Missing[0].Name == "StateDone" {
			sw = true
		}
		if f.Owner == "mapHole" && f.Rule == RuleLiteral && strings.Contains(f.Msg, "map") {
			mp = true
		}
		if f.Owner == "arrayHole" && f.Rule == RuleLiteral && f.Type == "Ordinal" {
			arr = true
		}
		if f.Owner == "switchDefault" {
			t.Fatalf("defaulted switch reported: %+v", f)
		}
	}
	if !sw || !mp || !arr {
		t.Fatalf("missing fixture findings switch=%v map=%v array=%v\n%s", sw, mp, arr, findingsText(rep.Findings))
	}
}

func TestReasonedExemptionAndStaleness(t *testing.T) {
	root := writeFixture(t, `package fixture
 type State string
 const (StateNew State = "new"; StateRunning State = "running"; StateDone State = "done")
 func partial(s State) { switch s { case StateNew, StateRunning: } }
 `)
	key := exemptKey(RuleSwitch, "internal/fixture", "partial")
	rep, err := Scan(root, Config{IncludeTestFiles: true, Exempt: func(got string) (string, bool) {
		if got == key {
			return "caller intentionally handles only new state", true
		}
		return "", false
	}})
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.Findings) != 0 {
		t.Fatalf("reasoned exemption not applied: %+v", rep)
	}
	rep, err = Scan(root, Config{IncludeTestFiles: true, Exempt: func(got string) (string, bool) {
		if got == key {
			return "", true
		}
		return "", false
	}})
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.Findings) != 1 || !strings.Contains(rep.Findings[0].Msg, "partial") {
		t.Fatalf("blank reason must not exempt: %+v", rep.Findings)
	}
}

func findingsText(fs []Finding) string {
	var b strings.Builder
	for _, f := range fs {
		b.WriteString(f.String())
		b.WriteByte('\n')
	}
	return b.String()
}
