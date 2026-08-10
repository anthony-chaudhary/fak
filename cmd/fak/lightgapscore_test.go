package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func lightgapRoot(t *testing.T) string {
	t.Helper()
	p, e := filepath.Abs(filepath.Join("..", ".."))
	if e != nil {
		t.Fatal(e)
	}
	return p
}
func runLightgap(t *testing.T, args ...string) (int, string, string) {
	t.Helper()
	var out, err bytes.Buffer
	args = append(args, "--workspace", lightgapRoot(t))
	code := runLightgapScore(&out, &err, args)
	return code, out.String(), err.String()
}
func TestLightgapJSONControlPaneContract(t *testing.T) {
	code, out, err := runLightgap(t, "--json")
	if code != 0 {
		t.Fatalf("code %d: %s", code, err)
	}
	var p map[string]any
	if e := json.Unmarshal([]byte(out), &p); e != nil {
		t.Fatal(e)
	}
	if _, ok := p["lightgap_debt"].(float64); !ok {
		t.Fatalf("control-pane debt missing: %#v", p["lightgap_debt"])
	}
	if p["schema"] != "fak-lightgap-scorecard/1" {
		t.Fatalf("schema=%v", p["schema"])
	}
}
func TestLightgapCheckRedsOnDebt(t *testing.T) {
	code, out, _ := runLightgap(t, "--check")
	if code != 1 || !strings.Contains(out, "lightgap_debt = 9") {
		t.Fatalf("code=%d out=%q", code, out)
	}
}
func TestLightgapTextModes(t *testing.T) {
	for _, tc := range [][]string{{}, {"--segment", "solo-max"}, {"--facet", "raw-speed"}, {"--dents"}, {"--unrun"}, {"--ceilings"}} {
		code, out, err := runLightgap(t, tc...)
		if code != 0 || out == "" {
			t.Fatalf("%v code=%d out=%q err=%q", tc, code, out, err)
		}
	}
}
func TestLightgapRejectsUnknownFacet(t *testing.T) {
	code, _, err := runLightgap(t, "--facet", "nope")
	if code != 2 || !strings.Contains(err, "unknown facet") {
		t.Fatalf("code=%d err=%q", code, err)
	}
}
func TestLightgapMarkdownDir(t *testing.T) {
	dir := t.TempDir()
	code, out, err := runLightgap(t, "--markdown-dir", dir)
	if code != 0 {
		t.Fatalf("code=%d err=%q", code, err)
	}
	if strings.Count(out, "wrote ") != 12 {
		t.Fatalf("out=%q", out)
	}
	if _, e := os.Stat(filepath.Join(dir, "README.md")); e != nil {
		t.Fatal(e)
	}
}
func TestLightgapScoreRouteRegistered(t *testing.T) {
	if scoreRoutes["lightgap"] == nil {
		t.Fatal("fak score lightgap not registered")
	}
}
