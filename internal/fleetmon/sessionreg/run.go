// Package sessionreg is the cross-account "what stopped, why, and how to resume" index: it
// classifies every recent Claude Code transcript under the user's .claude* config dirs,
// folds the account blockers they witness, decides which stopped sessions are safe to
// resume (and on which account), and publishes the fleet registry (sessions.json,
// resume_plan.json, decisions.log, transitions.log) the dispatch tick and the resume
// file formats byte-for-byte.
package sessionreg

import (
	"context"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/fleetaccounts"
)

// Modes are the report shapes Run renders.
const (
	ModeSummary  = "summary"
	ModeJSON     = "json"
	ModeRegistry = "registry"
	ModePlan     = "plan"
	ModeResume   = "resume"
)

// Pass is one completed classification + decision pass.
type Pass struct {
	o            Options
	Rows         []*Row
	Throttle     *Obj
	Auth         *Obj
	roster       []fleetaccounts.Account
	availability []availability
	ledger       ledgerFolds
}

// Evaluate scans, merges and decides without writing anything.
func Evaluate(ctx context.Context, o Options) (*Pass, error) {
	wc := fleetaccounts.NewWorkerCheck(o.Home, o.ConfigHome, o.Policy)
	cache := map[string]bool{}
	isWorker := func(acct string) bool {
		v, ok := cache[acct]
		if !ok {
			v = wc.IsWorker(acct)
			cache[acct] = v
		}
		return v
	}
	lf := readLedger(pyJoin(o.RegDir, "resume_ledger.jsonl"), o.MaxAttempts)
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	sr, err := scan(ctx, o, isWorker, lf.rearmed)
	if err != nil {
		return nil, err
	}
	prev := loadPrevRegistry(pyJoin(o.RegDir, "sessions.json"))
	p := &Pass{o: o, Rows: sr.rows, ledger: lf}
	p.Throttle = mergeKnownThrottle(o, sr.throttle, sr.rows, prev, isWorker)
	p.Auth = mergeKnownAuth(o, sr.rows, prev, isWorker)
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	p.roster = registryRoster(fleetaccounts.Discover(o.Home, o.ConfigHome, o.Policy), isWorker)
	p.availability = accountAvailability(o, p.roster, p.Throttle, p.Auth, p.Rows)
	var live map[string]bool
	if o.LiveSIDs != nil {
		live = o.LiveSIDs()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	decide(o, p.Rows, p.Throttle, p.availability, live, lf)
	return p, nil
}

// registryRoster is the seat set the registry publishes: the Claude and opencode workers
// the scan itself accepts. Codex homes are routed by the dispatch roster, never by a
// transcript re-home, and a seat the scan refuses is never offered as a re-home target.
func registryRoster(rows []fleetaccounts.Account, isWorker func(string) bool) []fleetaccounts.Account {
	out := make([]fleetaccounts.Account, 0, len(rows))
	for _, a := range rows {
		if a.Kind == fleetaccounts.KindWorker && ((a.Product != "claude" && a.Product != "opencode") || !isWorker(a.Account)) {
			continue
		}
		out = append(out, a)
	}
	return out
}

// accounts is the availability snapshot as of the decided rows (BLOCKED_AUTH actions now
// count toward an account's auth-blocked sessions).
func (p *Pass) accounts() []availability {
	return accountAvailability(p.o, p.roster, p.Throttle, p.Auth, p.Rows)
}

// RegistrySummary is the machine-readable result of a registry write, the object the
// dispatch tick records as its registry_refresh provenance.
type RegistrySummary struct {
	Sessions   int
	AutoResume int
	Probed     int
	Wrote      []string
}

// Map is the summary in the {"ok","mode","sessions","auto_resume","probed","wrote"} shape.
func (s RegistrySummary) Map() map[string]any {
	wrote := make([]any, len(s.Wrote))
	for i, w := range s.Wrote {
		wrote[i] = w
	}
	return map[string]any{"ok": true, "mode": ModeRegistry, "sessions": s.Sessions,
		"auto_resume": s.AutoResume, "probed": s.Probed, "wrote": wrote}
}

func (s RegistrySummary) obj() *Obj {
	e := NewObj()
	e.Set("ok", true)
	e.Set("mode", ModeRegistry)
	e.Set("sessions", s.Sessions)
	e.Set("auto_resume", s.AutoResume)
	e.Set("probed", s.Probed)
	e.Set("wrote", s.Wrote)
	return e
}

// WriteRegistry publishes sessions.json, resume_plan.json and the decision logs. A ctx that
// has already ended writes nothing, so a pass abandoned at its deadline cannot overwrite a
// newer registry later.
func (p *Pass) WriteRegistry(ctx context.Context) (RegistrySummary, error) {
	o := p.o
	if err := ctx.Err(); err != nil {
		return RegistrySummary{}, err
	}
	if err := os.MkdirAll(o.RegDir, 0o755); err != nil {
		return RegistrySummary{}, err
	}
	srows := sessionRows(p.Rows)
	sessions := make([]any, len(srows))
	for i, r := range srows {
		sessions[i] = registrySessionObj(r)
	}
	reg := NewObj()
	reg.Set("schema", "fleet-sessions/3")
	reg.Set("app_version", o.AppVersion)
	reg.Set("generated_utc", isoUTC(o.Now))
	reg.Set("window_h", o.WindowH)
	reg.Set("throttle", p.Throttle)
	reg.Set("auth", p.Auth)
	reg.Set("accounts", availabilityObjs(p.accounts()))
	reg.Set("sessions", sessions)
	reg.Set("summary", stormSummary(srows, p.Throttle))
	sp := pyJoin(o.RegDir, "sessions.json")
	pp := pyJoin(o.RegDir, "resume_plan.json")
	plan := o.planDoc(p.Rows)
	if err := writeText(sp, dumpPy(reg, 1)); err != nil {
		return RegistrySummary{}, err
	}
	if err := writeText(pp, dumpPy(plan, 1)); err != nil {
		return RegistrySummary{}, err
	}
	if err := o.logDecisions(p.Rows); err != nil {
		return RegistrySummary{}, err
	}
	planRows, _ := plan.Val("plan").([]any)
	return RegistrySummary{Sessions: len(srows), AutoResume: len(planRows), Wrote: []string{sp, pp}}, nil
}

// Run renders one mode to w (the CLI's stdout), writing the registry files in registry mode.
func Run(ctx context.Context, o Options, mode string, w io.Writer) (RegistrySummary, error) {
	p, err := Evaluate(ctx, o)
	if err != nil {
		return RegistrySummary{}, err
	}
	var b strings.Builder
	var sum RegistrySummary
	switch mode {
	case ModeJSON:
		rows := make([]any, len(p.Rows))
		for i, r := range p.Rows {
			rows[i] = rowObj(r)
		}
		doc := NewObj()
		doc.Set("app_version", o.AppVersion)
		doc.Set("now", isoUTC(o.Now))
		doc.Set("throttle", p.Throttle)
		doc.Set("auth", p.Auth)
		doc.Set("accounts", availabilityObjs(p.accounts()))
		doc.Set("probes", []any{})
		doc.Set("rows", rows)
		b.WriteString(dumpPy(doc, 1) + "\n")
	case ModeRegistry:
		sum, err = p.WriteRegistry(ctx)
		if err != nil {
			return sum, err
		}
		fmt.Fprintf(&b, "wrote %s (%d sessions)\n", sum.Wrote[0], sum.Sessions)
		fmt.Fprintf(&b, "wrote %s (%d AUTO_RESUME)\n", sum.Wrote[1], sum.AutoResume)
		b.WriteString(dumpPy(sum.obj(), -1) + "\n")
	case ModePlan:
		b.WriteString(dumpPy(o.planDoc(p.Rows), 1) + "\n")
	case ModeResume:
		p.renderResume(&b)
	default:
		p.renderSummary(&b)
	}
	_, err = io.WriteString(w, textMode(b.String()))
	return sum, err
}

func (p *Pass) renderAccounts(b *strings.Builder) {
	var avail, blocked []string
	for _, a := range p.accounts() {
		if a.Available {
			avail = append(avail, a.Tag)
		} else {
			blocked = append(blocked, a.Tag+" ("+a.BlockReason+")")
		}
	}
	b.WriteString("ACCOUNTS AVAILABLE NOW (worker, not blocked):\n")
	if len(avail) > 0 {
		b.WriteString("  " + strings.Join(avail, ", ") + "\n")
	} else {
		b.WriteString("  (none - all blocked)\n")
	}
	if len(blocked) > 0 {
		b.WriteString("BLOCKED: " + strings.Join(blocked, ", ") + "\n")
	}
	b.WriteString("\n")
}

// The resume headers keep the Python source's literal characters (an em-dash that was
// saved double-encoded), so the two writers print the same text.
const emDashMojibake = "â€”"

func (p *Pass) renderResume(b *strings.Builder) {
	p.renderAccounts(b)
	b.WriteString("# AUTO-RESUMABLE (autonomous, dead, account available) " + emDashMojibake + " safe to run:\n\n")
	n := 0
	for _, r := range p.Rows {
		if r.Action != "AUTO_RESUME" {
			continue
		}
		n++
		fmt.Fprintf(b, "# [%s] %s (%s)  age=%sm\n", r.Disp, r.Project, pyStr(jsonValue(r.Git)), pyFloatRepr(r.AgeMin))
		b.WriteString(pyStr(r.resumeCmdValue()) + "\n\n")
	}
	if n == 0 {
		b.WriteString("# (none right now)\n\n")
	}
	b.WriteString("# SURFACE " + emDashMojibake + " stopped but interactive; resume only if you mean to:\n\n")
	for _, r := range p.Rows {
		if r.Action != "SURFACE" {
			continue
		}
		fmt.Fprintf(b, "# [%s] %s age=%sm  -- %s\n", r.Disp, r.Project, pyFloatRepr(r.AgeMin), runeHead(r.Last, 70))
		b.WriteString(pyStr(r.resumeCmdValue()) + "\n\n")
	}
}

var dispOrder = []string{"STOPPED_LIMIT", "STOPPED_APIERR", "STOPPED_CONTEXT_EXHAUSTED",
	"INFRA_AUTH", "INFRA_ORG_DISABLED", "DEAD_MIDTOOL", "DEAD_KILLED", "NEVER_STARTED",
	"USER_CLOSED", "STOPPED_QUIET", "PARKED_WAIT", "LIVE", "DONE"}

var actionMarks = map[string]string{
	"AUTO_RESUME": " *AUTO", "SURFACE": " surface", "SUPERVISED": " (sup)",
	"ESCALATE_FRESH_CONTINUATION": " fresh-cont",
	"DEFER_THROTTLED":             " defer", "DEFER_NO_USAGE": " defer-no-usage",
	"DEFER_DUPLICATE_TASK": " dup-task", "BLOCKED_AUTH": " blocked-auth",
	"SKIP_DONE": " done", "SKIP_USER_CLOSED": " user-closed",
}

const summaryGroupCap = 40

func (p *Pass) renderSummary(b *strings.Builder) {
	o := p.o
	fmt.Fprintf(b, "fleet_sessions @ %sZ   window=%sh   %d sessions\n\n", o.Now.UTC().Format("2006-01-02 15:04"), pyFloatRepr(o.WindowH), len(p.Rows))
	p.renderAccounts(b)
	if p.Throttle.Len() > 0 {
		b.WriteString("THROTTLED ACCOUNTS (resume will instantly re-die):\n")
		for _, a := range p.Throttle.Keys() {
			t, _ := p.Throttle.Val(a).(*Obj)
			line := "  " + padRight(a, 30) + " resets " + pyStr(t.Val("reset"))
			if wk := t.Val("weekly"); truthy(wk) {
				line += "  | weekly " + pyStr(wk)
			}
			b.WriteString(line + "\n")
		}
		b.WriteString("\n")
	}
	counts, acts, cats := NewObj(), NewObj(), map[string]int{}
	nAgent := 0
	for _, r := range p.Rows {
		c, _ := counts.Val(r.Disp).(int)
		counts.Set(r.Disp, c+1)
		a, _ := acts.Val(r.Action).(int)
		acts.Set(r.Action, a+1)
		cats[r.Category]++
		if r.InitiatedBy == "agent" {
			nAgent++
		}
	}
	order := append([]string(nil), dispOrder...)
	known := map[string]bool{}
	for _, d := range order {
		known[d] = true
	}
	for _, d := range counts.Keys() {
		if !known[d] {
			order = append(order, d)
		}
	}
	var parts []string
	for _, k := range []string{"INFRA", "AGENT", "USER", "HANGING", "LIVE"} {
		if cats[k] > 0 {
			parts = append(parts, fmt.Sprintf("%s=%d", k, cats[k]))
		}
	}
	b.WriteString("category: " + strings.Join(parts, "  ") + "\n")
	parts = parts[:0]
	for _, k := range order {
		if c, _ := counts.Val(k).(int); c > 0 {
			parts = append(parts, fmt.Sprintf("%s=%d", k, c))
		}
	}
	b.WriteString("disp:     " + strings.Join(parts, "  ") + "\n")
	actKeys := append([]string(nil), acts.Keys()...)
	sort.Strings(actKeys)
	parts = parts[:0]
	for _, k := range actKeys {
		parts = append(parts, fmt.Sprintf("%s=%d", k, acts.Val(k).(int)))
	}
	b.WriteString("action:   " + strings.Join(parts, "  ") + "\n")
	fmt.Fprintf(b, "initiated: agent=%d  user=%d\n", nAgent, len(p.Rows)-nAgent)
	st, _ := stormSummary(sessionRows(p.Rows), p.Throttle).Val("storm").(*Obj)
	rate := st.Val("apierr_per_min_30m").(float64)
	flag := ""
	if rate >= 1.0 {
		flag = "  <-- STORM"
	}
	fmt.Fprintf(b, "storm:     apierr 15m=%d 30m=%d 60m=%d  rate=%s/min  frac30m=%s%s\n",
		st.Val("apierr_15m"), st.Val("apierr_30m"), st.Val("apierr_60m"), pyFloatRepr(rate),
		pyFloatRepr(st.Val("apierr_frac_30m").(float64)), flag)
	nsRate := st.Val("neverstart_per_min_30m").(float64)
	nsFlag := ""
	if nsRate >= 1.0 {
		nsFlag = "  <-- NEVER-START BURST"
	}
	fmt.Fprintf(b, "neverstart:      15m=%d 30m=%d 60m=%d  rate=%s/min%s\n\n",
		st.Val("neverstart_15m"), st.Val("neverstart_30m"), st.Val("neverstart_60m"), pyFloatRepr(nsRate), nsFlag)
	for _, disp := range order {
		var grp []*Row
		for _, r := range p.Rows {
			if r.Disp == disp {
				grp = append(grp, r)
			}
		}
		if len(grp) == 0 {
			continue
		}
		fmt.Fprintf(b, "== %s: %d ==\n", disp, len(grp))
		for i, r := range grp {
			if i == summaryGroupCap {
				break
			}
			mark := actionMarks[r.Action]
			if r.Rehomed {
				mark += " -> " + accountTag(r.ResumeAccount)
			}
			thr := ""
			if _, ok := p.Throttle.Get(r.Account); ok {
				thr = "  [THROTTLED]"
			}
			who := "u"
			if r.InitiatedBy == "agent" {
				who = "A"
			}
			code := ""
			if r.HTTPStatus != "" {
				code = " [" + r.HTTPStatus + "]"
			}
			relaunch := ""
			if n := p.ledger.inplace[r.Session]; n > 0 {
				relaunch = fmt.Sprintf("  relaunch×%d", n)
			}
			fmt.Fprintf(b, "  %sm %s %s %s %s%s%s%s%s\n", padLeft(pyFloatRepr(r.AgeMin), 6), who,
				padRight(accountTag(r.Account), 18), padRight(r.Project, 26), runeHead(r.Session, 8), mark, thr, code, relaunch)
		}
		if len(grp) > summaryGroupCap {
			fmt.Fprintf(b, "  ... +%d more\n", len(grp)-summaryGroupCap)
		}
		b.WriteString("\n")
	}
}
