package sessionreg

import (
	"fmt"
	"os"
	"runtime"
	"sort"
	"strings"
)

// newline is what Python's text-mode writes turn "\n" into on this platform; the registry
// files and the human report keep the bytes the Python writer produced.
var newline = func() string {
	if runtime.GOOS == "windows" {
		return "\r\n"
	}
	return "\n"
}()

func textMode(s string) string {
	if newline == "\n" {
		return s
	}
	return strings.ReplaceAll(s, "\n", newline)
}

func nilIfEmpty(s string) any {
	if s == "" {
		return nil
	}
	return s
}

// rowObj is a row as the json mode prints it: every field, in the key order the Python
// dict accumulated them.
func rowObj(r *Row) *Obj {
	e := NewObj()
	e.Set("disp", r.Disp)
	e.Set("category", r.Category)
	e.Set("cause", r.Cause)
	e.Set("reason", r.Reason)
	e.Set("last_kind", nilIfEmpty(r.LastKind))
	e.Set("age_min", r.AgeMin)
	e.Set("seen_utc", r.SeenUTC)
	e.Set("last_ts", r.LastTS)
	e.Set("throttle_reset", nilIfEmpty(r.ThrottleReset))
	e.Set("throttle_weekly", nilIfEmpty(r.ThrottleWeekly))
	e.Set("throttle_seen", nilIfEmpty(r.ThrottleSeen))
	e.Set("throttle_current", r.ThrottleCurrent)
	e.Set("pending_tool", jsonValue(r.PendingTool))
	e.Set("http_status", nilIfEmpty(r.HTTPStatus))
	e.Set("initiated_by", r.InitiatedBy)
	e.Set("session", r.Session)
	e.Set("cwd", jsonValue(r.Cwd))
	e.Set("git", jsonValue(r.Git))
	e.Set("last", r.Last)
	e.Set("path", r.Path)
	e.Set("autonomous", r.Autonomous)
	e.Set("supervised", r.Supervised)
	e.Set("task_sig", r.TaskSig)
	e.Set("records", r.Records)
	e.Set("account", r.Account)
	e.Set("project", r.Project)
	if r.actionEarly {
		e.Set("action", r.Action)
	}
	if r.decided {
		e.Set("source_config_dir", r.SourceConfigDir)
		e.Set("resume_account", r.ResumeAccount)
		e.Set("resume_config_dir", r.ResumeConfigDir)
		e.Set("rehomed", r.Rehomed)
		e.Set("action", r.Action)
		e.Set("resume_cmd", r.resumeCmdValue())
	}
	return e
}

func (r *Row) resumeCmdValue() any {
	if !r.hasResumeCmd {
		return nil
	}
	return r.ResumeCmd
}

// jsonValue maps a value decoded by encoding/json onto the ordered encoder's shapes.
func jsonValue(v any) any {
	switch t := v.(type) {
	case map[string]any:
		keys := make([]string, 0, len(t))
		for k := range t {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		o := NewObj()
		for _, k := range keys {
			o.Set(k, jsonValue(t[k]))
		}
		return o
	case []any:
		out := make([]any, len(t))
		for i, e := range t {
			out[i] = jsonValue(e)
		}
		return out
	case float64:
		if t == float64(int64(t)) && t < 1e15 && t > -1e15 {
			return int64(t)
		}
	}
	return v
}

func registrySessionObj(r *Row) *Obj {
	e := NewObj()
	e.Set("account", r.Account)
	e.Set("project", r.Project)
	e.Set("session", r.Session)
	e.Set("cwd", jsonValue(r.Cwd))
	e.Set("git", jsonValue(r.Git))
	e.Set("category", r.Category)
	e.Set("cause", r.Cause)
	e.Set("disp", r.Disp)
	e.Set("reason", r.Reason)
	e.Set("action", r.Action)
	e.Set("autonomous", r.Autonomous)
	e.Set("supervised", r.Supervised)
	e.Set("age_min", r.AgeMin)
	e.Set("seen_utc", r.SeenUTC)
	e.Set("throttle_reset", nilIfEmpty(r.ThrottleReset))
	e.Set("throttle_weekly", nilIfEmpty(r.ThrottleWeekly))
	e.Set("resume_cmd", r.resumeCmdValue())
	e.Set("rehomed", r.Rehomed)
	e.Set("resume_account", r.ResumeAccount)
	e.Set("last", r.Last)
	e.Set("task_sig", r.TaskSig)
	e.Set("http_status", nilIfEmpty(r.HTTPStatus))
	e.Set("initiated_by", r.InitiatedBy)
	return e
}

func (o Options) planEntry(r *Row) *Obj {
	e := NewObj()
	e.Set("account", r.Account)
	e.Set("config_dir", o.configDir(r.Account))
	e.Set("source_config_dir", r.SourceConfigDir)
	e.Set("resume_account", r.ResumeAccount)
	e.Set("resume_config_dir", r.ResumeConfigDir)
	e.Set("rehomed", r.Rehomed)
	e.Set("session", r.Session)
	e.Set("cwd", jsonValue(r.Cwd))
	e.Set("project", r.Project)
	e.Set("disp", r.Disp)
	e.Set("resume_cmd", r.resumeCmdValue())
	return e
}

func (o Options) planObjs(rows []*Row) []any {
	plan := []any{}
	for _, r := range rows {
		if r.Action == "AUTO_RESUME" {
			plan = append(plan, o.planEntry(r))
		}
	}
	return plan
}

func (o Options) planDoc(rows []*Row) *Obj {
	doc := NewObj()
	doc.Set("app_version", o.AppVersion)
	doc.Set("generated_utc", isoUTC(o.Now))
	doc.Set("plan", o.planObjs(rows))
	return doc
}

var stormBuckets = []int{15, 30, 60}

// stormSummary pre-aggregates the registry: counts per disposition, the recency-bucketed
// API-error and never-start storm rates, and per-account health.
func stormSummary(rows []*Row, throttle *Obj) *Obj {
	counts := NewObj()
	apierr, neverstart, total := map[int]int{}, map[int]int{}, map[int]int{}
	accts := NewObj()
	type health struct {
		newestDisp string
		newestAge  *float64
		apierr30   int
		live       int
	}
	hs := map[string]*health{}
	for _, r := range rows {
		d := r.Disp
		if d == "" {
			d = "?"
		}
		n, _ := counts.Val(d).(int)
		counts.Set(d, n+1)
		a := r.AgeMin
		if a < 0 {
			a = 0
		}
		isErr, isNS := d == "STOPPED_APIERR", d == "NEVER_STARTED"
		for _, w := range stormBuckets {
			if a <= float64(w) {
				total[w]++
				if isErr {
					apierr[w]++
				}
				if isNS {
					neverstart[w]++
				}
			}
		}
		if r.Account == "" {
			continue
		}
		h, ok := hs[r.Account]
		if !ok {
			h = &health{}
			hs[r.Account] = h
			accts.Set(r.Account, nil)
		}
		if h.newestAge == nil || a < *h.newestAge {
			v := a
			h.newestAge, h.newestDisp = &v, d
		}
		if isErr && a <= 30 {
			h.apierr30++
		}
		if d == "LIVE" {
			h.live++
		}
	}
	storm := NewObj()
	for _, w := range stormBuckets {
		storm.Set(fmt.Sprintf("apierr_%dm", w), apierr[w])
	}
	for _, w := range stormBuckets {
		storm.Set(fmt.Sprintf("total_%dm", w), total[w])
	}
	for _, w := range stormBuckets {
		storm.Set(fmt.Sprintf("neverstart_%dm", w), neverstart[w])
	}
	storm.Set("apierr_per_min_30m", pyRound(float64(apierr[30])/30.0, 3))
	frac := 0.0
	if total[30] > 0 {
		frac = pyRound(float64(apierr[30])/float64(total[30]), 3)
	}
	storm.Set("apierr_frac_30m", frac)
	storm.Set("neverstart_per_min_30m", pyRound(float64(neverstart[30])/30.0, 3))
	for _, acc := range accts.Keys() {
		h := hs[acc]
		e := NewObj()
		e.Set("newest_disp", nilIfEmpty(h.newestDisp))
		if h.newestAge == nil {
			e.Set("newest_age_min", nil)
		} else {
			e.Set("newest_age_min", *h.newestAge)
		}
		e.Set("apierr_30m", h.apierr30)
		e.Set("live", h.live)
		t := throttle.Val(acc)
		e.Set("throttled", truthy(t))
		var reset any
		if truthy(t) {
			if m, ok := t.(*Obj); ok {
				reset = m.Val("reset")
			} else {
				reset = t
			}
		}
		e.Set("throttle_reset", reset)
		accts.Set(acc, e)
	}
	out := NewObj()
	out.Set("counts_by_disp", counts)
	out.Set("storm", storm)
	out.Set("accounts", accts)
	return out
}

func sessionRows(rows []*Row) []*Row {
	var out []*Row
	for _, r := range rows {
		if r.Project != "_probe" {
			out = append(out, r)
		}
	}
	return out
}

func accountTag(acct string) string {
	return strings.ReplaceAll(strings.ReplaceAll(acct, ".claude-", ""), ".claude", "default")
}

func writeText(path, body string) error {
	return os.WriteFile(path, []byte(textMode(body)), 0o644)
}

// logDecisions writes the decisions.log snapshot, appends disposition changes since the
// previous pass to transitions.log, and records this pass's dispositions for the next one.
func (o Options) logDecisions(rows []*Row) error {
	now := isoUTC(o.Now)
	cats := map[string]int{}
	for _, r := range rows {
		cats[r.Category]++
	}
	catKeys := make([]string, 0, len(cats))
	for k := range cats {
		catKeys = append(catKeys, k)
	}
	sort.Strings(catKeys)
	parts := make([]string, len(catKeys))
	for i, k := range catKeys {
		parts[i] = fmt.Sprintf("%s=%d", k, cats[k])
	}
	var b strings.Builder
	fmt.Fprintf(&b, "# fleet session decisions @ %s  (%d sessions)\n", now, len(rows))
	b.WriteString("# categories: " + strings.Join(parts, "  ") + "\n")
	b.WriteString("# age    category project                    disp/action            cause / reason  [sid]\n")
	sorted := append([]*Row(nil), rows...)
	sort.SliceStable(sorted, func(i, j int) bool {
		if sorted[i].Category != sorted[j].Category {
			return sorted[i].Category < sorted[j].Category
		}
		return sorted[i].AgeMin < sorted[j].AgeMin
	})
	for _, r := range sorted {
		fmt.Fprintf(&b, "%sm %s %s %s %s %s %s  [%s/%s]\n", padLeft(pyFloatRepr(r.AgeMin), 7),
			padRight(r.Category, 8), padRight(r.Project, 26), padRight(r.Disp, 14),
			padRight(r.Action, 16), padRight(r.Cause, 16), r.Reason, accountTag(r.Account), runeHead(r.Session, 8))
	}
	if err := writeText(pyJoin(o.RegDir, "decisions.log"), b.String()); err != nil {
		return err
	}

	prevPath := pyJoin(o.RegDir, "_prev_disp.json")
	prev := NewObj()
	if data, err := os.ReadFile(prevPath); err == nil {
		if v, derr := decodeOrdered(data); derr == nil {
			if m, ok := v.(*Obj); ok {
				prev = m
			}
		}
	}
	cur := NewObj()
	bySID := map[string]*Row{}
	for _, r := range rows {
		cur.Set(r.Session, r.Disp)
		bySID[r.Session] = r
	}
	var t strings.Builder
	for _, sid := range cur.Keys() {
		d := cur.Val(sid).(string)
		old := prev.Val(sid)
		if !truthy(old) {
			continue
		}
		if s, ok := old.(string); ok && s == d {
			continue
		}
		r := bySID[sid]
		fmt.Fprintf(&t, "%s  [%s] %s  %s/%s  %s -> %s  [%s]  %s\n", now, padRight(r.Category, 7),
			runeHead(sid, 8), accountTag(r.Account), r.Project, pyStr(old), d, r.Action, r.Reason)
	}
	f, err := os.OpenFile(pyJoin(o.RegDir, "transitions.log"), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	_, werr := f.WriteString(textMode(t.String()))
	if cerr := f.Close(); werr == nil {
		werr = cerr
	}
	if werr != nil {
		return werr
	}
	return writeText(prevPath, dumpPy(cur, -1))
}
