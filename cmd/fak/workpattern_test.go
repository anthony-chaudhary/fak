package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func captureWorkpattern(t *testing.T, args ...string) string {
	t.Helper()
	old := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w
	var b bytes.Buffer
	done := make(chan struct{})
	go func() { _, _ = b.ReadFrom(r); close(done) }()
	err := cmdWorkpattern(args)
	w.Close()
	os.Stdout = old
	<-done
	r.Close()
	if err != nil {
		t.Fatalf("cmdWorkpattern: %v", err)
	}
	return b.String()
}
func TestWorkpatternListJSONDeterministic(t *testing.T) {
	a := captureWorkpattern(t, "list", "--json")
	b := captureWorkpattern(t, "list", "--json")
	if a != b {
		t.Fatal("nondeterministic list")
	}
	var r workpatternReport
	if err := json.Unmarshal([]byte(a), &r); err != nil {
		t.Fatal(err)
	}
	if r.Schema != workpatternReportSchema || r.Catalog == nil || len(r.Catalog.Patterns) < 8 || len(r.Catalog.Subpatterns) < 12 {
		t.Fatalf("bad report %#v", r)
	}
}
func TestWorkpatternSourceAndPrivacy(t *testing.T) {
	root := t.TempDir()
	src := `package x
import "os"
func flow(){ b,_:=os.ReadFile("x"); _=os.WriteFile("x",b,0600); if len(b)==0 { panic("bad") } }
`
	if err := os.WriteFile(filepath.Join(root, "x.go"), []byte(src), 0600); err != nil {
		t.Fatal(err)
	}
	a := captureWorkpattern(t, "source", "--source", root, "--json")
	b := captureWorkpattern(t, "source", "--source", root, "--json")
	if a != b {
		t.Fatal("source output changed")
	}
	if !strings.Contains(a, "sp.inspect-edit-verify") {
		t.Fatalf("missing motif %s", a)
	}
	chat := filepath.Join(root, "chat.json")
	secret := "SECRET_SHOULD_NOT_LEAK"
	body := `{"format":"fak.scrubbed-chat/1","raw_message":"` + secret + `","conversations":[{"id":"c1","entries":[{"role":"assistant","tool":"Read","status":"ok"},{"role":"assistant","tool":"Edit","status":"ok"},{"role":"assistant","tool":"Bash","status":"ok"}]}]}`
	if err := os.WriteFile(chat, []byte(body), 0600); err != nil {
		t.Fatal(err)
	}
	out := captureWorkpattern(t, "trajectory", "--chat", chat, "--json")
	if strings.Contains(out, secret) {
		t.Fatal("default report leaked chat content")
	}
	if !strings.Contains(out, "sp.read-edit-verify") {
		t.Fatalf("missing trajectory finding %s", out)
	}
}
func TestWorkpatternBadArguments(t *testing.T) {
	for _, a := range [][]string{{"bogus"}, {"source"}, {"trajectory"}, {"trajectory", "--chat", "a", "--trajectory", "b"}} {
		if err := cmdWorkpattern(a); err == nil {
			t.Fatalf("args %v accepted", a)
		}
	}
}
