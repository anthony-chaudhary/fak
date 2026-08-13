package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/portability"
)

func runContinuitySelfcheck(stdout, stderr io.Writer, jsonOut bool) int {
	root, e := os.MkdirTemp("", "fak-continuity-selfcheck-")
	if e != nil {
		return 1
	}
	defer os.RemoveAll(root)
	a, b := filepath.Join(root, "home-a"), filepath.Join(root, "home-b")
	fixtures := map[string]string{"skills/review.json": `{"behavior":"review-concisely","summary":"Concise reviewer"}`, "workflows/triage.json": `{"behavior":"triage-before-fix","steps":["inspect","fix"]}`, "policies/safe.json": `{"behavior":"deny-destructive","allow":["read"]}`}
	for p, v := range fixtures {
		path := filepath.Join(a, "managed", p)
		os.MkdirAll(filepath.Dir(path), 0700)
		os.WriteFile(path, []byte(v), 0600)
	}
	sa, sb := portability.New(a), portability.New(b)
	pkgPath := filepath.Join(root, "context.fakpkg.json")
	p, er, err := sa.Export(pkgPath, nil, true)
	if err != nil {
		return selfcheckFail(stderr, err)
	}
	ar, err := sb.Apply(pkgPath, true, 0)
	if err != nil {
		return selfcheckFail(stderr, err)
	}
	sr, err := sb.Switch(p.ID, true)
	if err != nil {
		return selfcheckFail(stderr, err)
	}
	behavior, err := sb.Readback()
	if err != nil || len(behavior) != 3 {
		return selfcheckFail(stderr, fmt.Errorf("behavior readback: %v (%d objects)", err, len(behavior)))
	}
	rr, err := sb.Rollback(sr.ID, true)
	if err != nil {
		return selfcheckFail(stderr, err)
	}
	active, _ := sb.Active()
	if active != "" {
		return selfcheckFail(stderr, fmt.Errorf("rollback left %q active", active))
	}
	out := map[string]any{"result": "PASS", "service": "none", "homes": 2, "objects": 3, "package_id": p.ID, "receipts": map[string]string{"export": er.ID, "apply": ar.ID, "switch": sr.ID, "rollback": rr.ID}, "behavior": behavior, "rollback_active": active}
	if jsonOut {
		j, _ := json.MarshalIndent(out, "", "  ")
		fmt.Fprintln(stdout, string(j))
	} else {
		fmt.Fprintf(stdout, "PASS personal continuity: 3 real objects, 2 isolated homes, no service\npackage %s\nbehavior skill=%s workflow=%s policy=%s\nreceipts export=%s apply=%s switch=%s rollback=%s\nrollback restored prior inactive context\n", p.ID, behavior["skill:review"], behavior["workflow:triage"], behavior["policy:safe"], er.ID, ar.ID, sr.ID, rr.ID)
	}
	return 0
}
func selfcheckFail(w io.Writer, err error) int {
	fmt.Fprintf(w, "FAIL personal continuity: %v\n", err)
	return 1
}
