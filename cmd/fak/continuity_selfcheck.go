package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

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
	base := portability.Package{Schema: portability.Schema, Objects: []portability.Object{selfcheckObject("skill:sync", `{"owner":{"sensitivity":"public","value":"base"},"mode":{"sensitivity":"public","value":"base"}}`)}}
	base.Digest = selfcheckPackageDigest(base)
	local := portability.Package{Schema: portability.Schema, Objects: []portability.Object{selfcheckObject("skill:sync", `{"owner":{"sensitivity":"public","value":"home-a"},"mode":{"sensitivity":"public","value":"base"}}`)}}
	local.Digest = selfcheckPackageDigest(local)
	remote := portability.Package{Schema: portability.Schema, Objects: []portability.Object{selfcheckObject("skill:sync", `{"owner":{"sensitivity":"public","value":"base"},"mode":{"sensitivity":"public","value":"home-b"}}`)}}
	remote.Digest = selfcheckPackageDigest(remote)
	merge, err := portability.PreviewMerge(&base, local, remote, portability.ChannelPublic)
	if err != nil || len(merge.Conflicts) != 0 {
		return selfcheckFail(stderr, fmt.Errorf("offline merge: %v conflicts=%d", err, len(merge.Conflicts)))
	}
	planPath, mergedPath := filepath.Join(root, "merge.plan.json"), filepath.Join(root, "merged.fakpkg.json")
	if err = portability.WriteMergePlan(planPath, merge); err != nil {
		return selfcheckFail(stderr, err)
	}
	mr, err := sb.CommitMerge(merge, mergedPath, true, 0)
	if err != nil {
		return selfcheckFail(stderr, err)
	}
	active, _ := sb.Active()
	if active != merge.Result.ID {
		return selfcheckFail(stderr, fmt.Errorf("merge activation=%q want %q", active, merge.Result.ID))
	}
	mergeRollback, err := sb.Rollback(mr.ID, true)
	if err != nil {
		return selfcheckFail(stderr, err)
	}
	active, _ = sb.Active()
	if active != "" {
		return selfcheckFail(stderr, fmt.Errorf("merge rollback left %q active", active))
	}
	out := map[string]any{"result": "PASS", "service": "none", "homes": 2, "objects": 3, "package_id": p.ID, "receipts": map[string]string{"export": er.ID, "apply": ar.ID, "switch": sr.ID, "rollback": rr.ID, "merge": mr.ID, "merge_rollback": mergeRollback.ID}, "behavior": behavior, "rollback_active": active, "merge_plan": merge.ID, "merge_steps": len(merge.Steps), "merge_conflicts": len(merge.Conflicts)}
	if jsonOut {
		j, _ := json.MarshalIndent(out, "", "  ")
		fmt.Fprintln(stdout, string(j))
	} else {
		fmt.Fprintf(stdout, "PASS offline multi-home continuity: deterministic three-way plan %s, conflicts=%d\nPASS personal continuity: 3 real objects, 2 isolated homes, no service\npackage %s\nbehavior skill=%s workflow=%s policy=%s\nreceipts export=%s apply=%s switch=%s rollback=%s\nrollback restored prior inactive context\n", merge.ID, len(merge.Conflicts), p.ID, behavior["skill:review"], behavior["workflow:triage"], behavior["policy:safe"], er.ID, ar.ID, sr.ID, rr.ID)
	}
	return 0
}
func selfcheckFail(w io.Writer, err error) int {
	fmt.Fprintf(w, "FAIL personal continuity: %v\n", err)
	return 1
}

func selfcheckObject(id string, payload string) portability.Object {
	parts := strings.SplitN(id, ":", 2)
	return portability.Object{ID: id, Kind: parts[0], Name: parts[1], Active: true, Payload: json.RawMessage(payload)}
}
func selfcheckPackageDigest(p portability.Package) string {
	b, _ := json.Marshal(p.Objects)
	h := sha256.Sum256(b)
	return "sha256:" + hex.EncodeToString(h[:])
}
