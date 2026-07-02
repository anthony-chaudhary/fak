package main

// Commit-time diff-witness binding for the dispatch tick (#1324 proposal #2), the Go
// port of witness_exited_workers in tools/issue_resolve_dispatch.py. For every
// resolve-<N>-<stamp>.log whose pid is provably DEAD and not yet witnessed (no
// .witness sidecar), find the commit it landed for its issue (subject cites #N inside
// the per-worker .basesha..HEAD window) and grade it through `dos commit-audit`: a
// diff-witnessed commit -> CLAIM_WITNESSED, an unwitnessed or wrong-issue commit ->
// CLAIM_UNWITNESSED, no resolving commit -> CLAIM_NO_COMMIT with a structured reason
// classified from the log tail. The verdict is recorded in a .witness sidecar on live
// ticks so a bare `exit 0` / non-empty log never SILENTLY counts as productive, and
// the pick holds the re-blockable guard refusals it surfaces (#1396). Dead-pid gated
// (a still-running worker may not have committed yet -- never mis-blame it) and
// FAIL-OPEN throughout, the same discipline as the live-lane reap.

import (
	"context"
	"encoding/json"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

// Injectable seams mirroring the Python sweep's git= / audit_runner= test params.
var dispatchWitnessResolvingSHA = dispatchWitnessResolvingSHAGit
var dispatchWitnessCommitAudit = dispatchWitnessCommitAuditDos

// dispatchWitnessScanLimit bounds the no-basesha fallback window, mirroring the
// Python worker_resolving_sha scan_limit.
const dispatchWitnessScanLimit = 300

func witnessExitedWorkers(root, runsDir string, live bool) (map[string]any, []dispatchtick.WitnessRecord) {
	audited := []any{}
	buckets := map[string][]any{
		dispatchtick.ClaimWitnessed:   {},
		dispatchtick.ClaimUnwitnessed: {},
		dispatchtick.ClaimNoCommit:    {},
	}
	var records []dispatchtick.WitnessRecord
	for _, log := range resolveLogs(runsDir) {
		issue, ok := issueFromResolveLog(filepath.Base(log))
		if !ok {
			continue
		}
		stem := strings.TrimSuffix(log, filepath.Ext(log))
		if _, err := os.Stat(stem + dispatchtick.WitnessSidecarSuffix); err == nil {
			continue // audited once; a commit's diff (so its verdict) is immutable
		}
		pid, ok := readPID(stem + ".pid")
		if !ok {
			continue // no pid -> cannot prove the worker finished -> not yet auditable
		}
		if dispatchPIDAlive(pid) {
			continue // still running -> it may not have committed yet
		}
		base := ""
		if b, err := os.ReadFile(stem + dispatchtick.BaseSHASidecarSuffix); err == nil {
			base = strings.TrimSpace(string(b))
		}
		sha := dispatchWitnessResolvingSHA(root, issue, base)
		var rec dispatchtick.WitnessRecord
		if sha == "" {
			tail, size := dispatchWitnessLogTail(log)
			rec = dispatchtick.WitnessRecord{
				Issue:  issue,
				Log:    filepath.Base(log),
				Claim:  dispatchtick.ClaimNoCommit,
				Reason: dispatchtick.ClassifyNoCommitReason(tail, size),
			}
		} else {
			verdict, witness := dispatchWitnessCommitAudit(root, sha)
			claim := dispatchtick.ClaimUnwitnessed
			if dispatchtick.CommitWitnessed(verdict, witness) {
				claim = dispatchtick.ClaimWitnessed
			}
			rec = dispatchtick.WitnessRecord{
				Issue:   issue,
				Log:     filepath.Base(log),
				SHA:     sha,
				Claim:   claim,
				Verdict: verdict,
				Witness: witness,
			}
		}
		records = append(records, rec)
		row := rec.Map()
		audited = append(audited, row)
		buckets[rec.Claim] = append(buckets[rec.Claim], row)
		if live {
			if b, err := json.Marshal(row); err == nil {
				_ = os.WriteFile(stem+dispatchtick.WitnessSidecarSuffix, b, 0o644)
			}
		}
	}
	payload := map[string]any{
		"live":        live,
		"audited":     audited,
		"witnessed":   buckets[dispatchtick.ClaimWitnessed],
		"unwitnessed": buckets[dispatchtick.ClaimUnwitnessed],
		"no_commit":   buckets[dispatchtick.ClaimNoCommit],
	}
	return payload, records
}

// dispatchWitnessLogTail reads the last WitnessTailBytes of a worker log (the guard
// summary + final turn live at the end) without loading a possibly multi-MB file.
// Fail-open: a stat error yields ("", -1) so the classifier's size floor disengages.
func dispatchWitnessLogTail(log string) (string, int64) {
	st, err := os.Stat(log)
	if err != nil {
		return "", -1
	}
	size := st.Size()
	f, err := os.Open(log)
	if err != nil {
		return "", size
	}
	defer f.Close()
	if size > dispatchtick.WitnessTailBytes {
		if _, err := f.Seek(-int64(dispatchtick.WitnessTailBytes), io.SeekEnd); err != nil {
			return "", size
		}
	}
	b, err := io.ReadAll(f)
	if err != nil {
		return "", size
	}
	return string(b), size
}

// dispatchWitnessResolvingSHAGit finds the newest commit whose SUBJECT cites #issue,
// scoped to baseSHA..HEAD (the per-worker window recorded at spawn) when the base is
// known, else the most recent dispatchWitnessScanLimit commits. Fail-open: any git
// error yields "" so the slot claims nothing.
func dispatchWitnessResolvingSHAGit(root string, issue int, baseSHA string) string {
	args := []string{"log", "--no-color", "--pretty=format:%H\x1f%s"}
	if baseSHA != "" {
		args = append(args, baseSHA+"..HEAD")
	} else {
		args = append(args, "-n", strconv.Itoa(dispatchWitnessScanLimit))
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "git", args...)
	cmd.Dir = root
	configureDispatchHelperCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	return dispatchtick.FirstResolvingSHA(string(out), issue)
}

// dispatchWitnessCommitAuditDos grades sha through `dos commit-audit --json` and
// returns its (verdict, witness) pair. The command emits a JSON array, one row per
// audited sha; a dict is accepted too. Fail-open: an exec/parse failure yields empty
// strings, which grade to the conservative CLAIM_UNWITNESSED.
func dispatchWitnessCommitAuditDos(root, sha string) (string, string) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "dos", "commit-audit", sha, "--workspace", root, "--json")
	cmd.Dir = root
	configureDispatchHelperCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return "", ""
	}
	var parsed any
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(out))), &parsed); err != nil {
		return "", ""
	}
	row := map[string]any{}
	switch doc := parsed.(type) {
	case map[string]any:
		row = doc
	case []any:
		for _, item := range doc {
			m, ok := item.(map[string]any)
			if !ok {
				continue
			}
			if rowSHA := dispatchMapString(m, "sha"); rowSHA != "" && strings.HasPrefix(sha, rowSHA) {
				row = m
				break
			}
		}
		if len(row) == 0 && len(doc) > 0 {
			if m, ok := doc[0].(map[string]any); ok {
				row = m
			}
		}
	}
	return dispatchMapString(row, "verdict"), dispatchMapString(row, "witness")
}
