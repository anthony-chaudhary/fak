package sessionreg

import (
	"bufio"
	"encoding/json"
	"os"
	"strings"
)

// ledgerFolds is everything the planner reads from resume_ledger.jsonl, folded in one pass:
// the sids whose latest attempt-budget event is a re-arm (they bypass the scan window), the
// sids permanently blocked from a dedup primary election, and each sid's in-place resume
// streak.
type ledgerFolds struct {
	rearmed  map[string]bool
	blocked  map[string]bool
	inplace  map[string]int
	launches map[string]int
}

func readLedger(path string, maxAttempts int) ledgerFolds {
	lf := ledgerFolds{rearmed: map[string]bool{}, blocked: map[string]bool{}, inplace: map[string]int{}, launches: map[string]int{}}
	f, err := os.Open(path)
	if err != nil {
		return lf
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 64*1024*1024)
	for sc.Scan() {
		line := pyStrip(validUTF8(sc.Bytes()))
		if line == "" {
			continue
		}
		var rec map[string]any
		if json.Unmarshal([]byte(line), &rec) != nil || rec == nil {
			continue
		}
		lf.foldRearm(rec)
		lf.foldBlocked(rec)
		lf.foldInplace(rec)
	}
	for sid, n := range lf.launches {
		if n >= maxAttempts {
			lf.blocked[sid] = true
		}
	}
	return lf
}

func recStr(rec map[string]any, k string) string {
	v := rec[k]
	if !truthy(v) {
		return ""
	}
	return pyStr(v)
}

func (lf ledgerFolds) foldRearm(rec map[string]any) {
	sid := strings.TrimSpace(recStr(rec, "session"))
	if sid == "" {
		return
	}
	phase := recStr(rec, "phase")
	if phase == "" {
		phase = recStr(rec, "outcome")
	}
	phase = strings.ToLower(strings.TrimSpace(phase))
	action := strings.ToLower(strings.TrimSpace(recStr(rec, "action")))
	switch {
	case phase == "rearm":
		lf.rearmed[sid] = true
	case phase == "launched" || phase == "settled" || phase == "operator_settled" || phase == "consolidated" || strings.HasPrefix(action, "consolidate"):
		delete(lf.rearmed, sid)
	}
}

func isRearm(rec map[string]any) bool {
	return rec["phase"] == "rearm" || rec["outcome"] == "rearm"
}

func countsAsLaunch(rec map[string]any) bool {
	p := rec["phase"]
	return p == "launched" || p == "resumed" || truthy(rec["cause"])
}

func (lf ledgerFolds) foldBlocked(rec map[string]any) {
	sid, ok := rec["session"].(string)
	if !ok || sid == "" {
		return
	}
	if isRearm(rec) {
		lf.launches[sid] = 0
		delete(lf.blocked, sid)
		return
	}
	action := ""
	if v, has := rec["action"]; has {
		action = pyStr(v)
	}
	if truthy(rec["manual_override"]) || strings.HasPrefix(action, "consolidate") {
		lf.blocked[sid] = true
	}
	if rec["outcome"] == "unrecoverable" {
		lf.blocked[sid] = true
	}
	if countsAsLaunch(rec) {
		lf.launches[sid]++
	}
}

func (lf ledgerFolds) foldInplace(rec map[string]any) {
	sid, ok := rec["session"].(string)
	if !ok || sid == "" || truthy(rec["rehomed"]) {
		return
	}
	if isRearm(rec) {
		lf.inplace[sid] = 0
		return
	}
	if countsAsLaunch(rec) {
		lf.inplace[sid]++
	}
}
