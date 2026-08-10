// Command archreportdemo demonstrates fak's enforced architecture graph without a key,
// network, Git checkout, or mutable repository state.
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"reflect"

	"github.com/anthony-chaudhary/fak/internal/archreport"
)

func main() {
	selfcheck := flag.Bool("selfcheck", false, "verify the deterministic architecture-report invariant")
	flag.Parse()
	if flag.NArg() != 0 {
		fmt.Fprintln(os.Stderr, "usage: go run ./cmd/archreportdemo [-selfcheck]")
		os.Exit(2)
	}
	output, err := runDemo(*selfcheck)
	if err != nil {
		fmt.Fprintf(os.Stderr, "archreportdemo: %v\n", err)
		os.Exit(1)
	}
	fmt.Print(output)
}

func runDemo(selfcheck bool) (string, error) {
	root, err := os.MkdirTemp("", "fak-archreportdemo-")
	if err != nil {
		return "", err
	}
	defer os.RemoveAll(root)
	if err := writeFixture(root); err != nil {
		return "", err
	}
	report, err := archreport.Analyze(root, "")
	if err != nil {
		return "", err
	}
	wantHotspots := []archreport.Hotspot{{Name: "abi", FanIn: 2}, {Name: "policy", FanIn: 2}}
	wantDiagnostic := archreport.Diagnostic{Kind: "stale-tier-declaration", Leaf: "retired", Recovery: "create the package or remove its stale tier declaration"}
	checks := []struct {
		name string
		ok   bool
	}{
		{"schema", report.Schema == "fak-architecture/1"},
		{"healthy-leaves", len(report.Leaves) == 4},
		{"tier-counts", len(report.Tiers) == 3},
		{"upward-violation", report.Violations == 1},
		{"fan-in-hotspots", reflect.DeepEqual(report.Hotspots, wantHotspots)},
		{"stale-diagnostic", len(report.Diagnostics) == 1 && report.Diagnostics[0].Kind == wantDiagnostic.Kind && report.Diagnostics[0].Leaf == wantDiagnostic.Leaf && report.Diagnostics[0].Recovery == wantDiagnostic.Recovery},
	}
	for _, check := range checks {
		if !check.ok {
			return "", fmt.Errorf("selfcheck %s failed: report=%+v", check.name, report)
		}
	}
	out := "fak architecture report demo\n" +
		"schema: fak-architecture/1\n" +
		"healthy leaves: 4 across 3 tiers\n" +
		"upward violations: 1 (primitive -> composite)\n" +
		"direct fan-in hotspots: abi=2, policy=2\n" +
		"diagnostic: retired has a stale tier declaration\n"
	if selfcheck {
		out += "selfcheck: PASS (real archreport seam, deterministic fixture)\n"
	}
	return out, nil
}

func writeFixture(root string) error {
	files := map[string]string{
		"internal/architest/architest_test.go": `package architest
var tier=map[string]int{"abi":0,"primitive":1,"policy":2,"service":2,"retired":2}
var tierName=[]string{"root","primitive","foundation-composite"}
`,
		"internal/abi/abi.go": "package abi\n",
		"internal/primitive/primitive.go": `package primitive
import (
 _ "github.com/anthony-chaudhary/fak/internal/abi"
 _ "github.com/anthony-chaudhary/fak/internal/policy"
)
`,
		"internal/policy/policy.go": `package policy
import _ "github.com/anthony-chaudhary/fak/internal/abi"
`,
		"internal/service/service.go": `package service
import _ "github.com/anthony-chaudhary/fak/internal/policy"
`,
	}
	for name, body := range files {
		path := filepath.Join(root, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			return err
		}
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			return err
		}
	}
	return nil
}
