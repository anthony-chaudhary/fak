package main

import (
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/guardtrace"
	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

// runGuardReplay replays a recorded trace fixture through the REAL guard end to end and
// prints, turn by turn, what the floor did — the observable, no-API-key, no-GPU way to
// watch `fak guard` fire on a trace that leads to token work. It is the operator-facing
// twin of the internal/gateway end-to-end test: SAME floor, SAME gateway adjudication
// path, SAME decision journal, SAME exit summary the live `fak guard -- claude` prints.
//
// It stands up the gateway pointed at a built-in fake upstream (internal/guardtrace) that
// emits the fixture's tool_use + token-usage turns, posts each turn through the gateway's
// real /v1/messages or /v1/chat/completions route, reads back the per-call verdicts from
// the `fak` extension, and renders a one-line-per-call report plus the per-turn token
// economy and the journal rows recorded. Returns a process exit code: 0 when every call
// landed on the disposition the fixture declared, 1 otherwise (so it doubles as a CI demo
// check), 2 on a setup error.
func runGuardReplay(fixturePath, wire, policyPath, auditPath string, noAudit bool, out io.Writer) int {
	provider := normalizeReplayWire(wire)
	if provider == "" {
		fmt.Fprintf(out, "fak guard --replay-trace: unknown --replay-wire %q (want anthropic|openai)\n", wire)
		return 2
	}

	f, err := guardtrace.LoadFixture(fixturePath)
	if err != nil {
		fmt.Fprintf(out, "fak guard --replay-trace: %v\n", err)
		return 2
	}

	// Install the SAME capability floor the live path installs: an explicit --policy file
	// wins, else the embedded shipped guard floor. This is the whole point — the replay
	// fires the production floor, not a toy.
	var (
		rt          policy.Runtime
		floorSource string
	)
	if policyPath != "" {
		rt, err = policy.LoadRuntime(policyPath)
		floorSource = policyPath
	} else {
		rt, err = policy.ParseRuntime(guardDefaultPolicyJSON)
		floorSource = "built-in guard floor"
	}
	if err != nil {
		fmt.Fprintf(out, "fak guard --replay-trace: load floor: %v\n", err)
		return 2
	}
	if grant, _ := applyLaunchToolGrant(&rt); len(grant.Allow) > 0 {
		floorSource += fmt.Sprintf(" + launch-scoped allow grant (%d exact tool(s))", len(grant.Allow))
	}
	adjudicator.Default.SetPolicy(rt.Adjudicator)
	applyRuntime(rt)

	// Turn the durable, hash-chained decision journal ON for the replay, so the report can
	// show the rows recorded and prove the chain verifies. Replay keeps its historical temp
	// path by default, but an operator-supplied --audit path must be honored exactly.
	var j *journal.Journal
	var jerr error
	jpath, auditOptedOut, auditRequested := guardReplayAuditPlan(auditPath, noAudit)
	if !auditOptedOut {
		j, jerr = journal.Enable(jpath)
		if jerr != nil {
			fmt.Fprintf(out, "fak guard --replay-trace: audit journal disabled: %v\n", jerr)
			if auditRequested {
				return 2
			}
		} else if auditRequested && j != nil && j.Path() != "" && !sameReplayAuditPath(j.Path(), jpath) {
			fmt.Fprintf(out, "fak guard --replay-trace: requested --audit %s but active journal is %s\n", jpath, j.Path())
			return 2
		}
	}

	const model = "fak-guard-replay:model"
	upstream := guardtrace.NewFakeUpstream(provider, model, f)
	defer upstream.Close()

	srv, err := gateway.New(gateway.Config{
		EngineID:             "mock",
		Model:                model,
		Provider:             provider,
		BaseURL:              replayUpstreamBase(provider, upstream.URL),
		VDSO:                 true,
		Invalidation:         "global",
		Version:              appversion.Current(),
		CtxViewBudget:        agent.DefaultCtxViewBudget,
		CompactHistoryBudget: gateway.DefaultCompactHistoryBudget,
		ElideResultBytes:     gateway.DefaultElideResultBytes,
		ToolFloorDenies:      adjudicator.Default.NeverAdmits,
		// Mute the structured per-request log stream so the human report stands alone (the
		// live `fak guard` keeps it off by default too); the durable journal + the report
		// carry the record.
		Logf: func(string, ...any) {},
	})
	if err != nil {
		fmt.Fprintf(out, "fak guard --replay-trace: gateway init: %v\n", err)
		return 2
	}

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		fmt.Fprintf(out, "fak guard --replay-trace: listen: %v\n", err)
		return 2
	}
	gwURL := "http://" + ln.Addr().String()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	serveErr := make(chan error, 1)
	go func() { serveErr <- srv.Serve(ctx, ln) }()
	if werr, consumed := guardWaitHealthy(gwURL, serveErr, 5*time.Second); werr != nil {
		cancel()
		if !consumed {
			<-serveErr
		}
		fmt.Fprintf(out, "fak guard --replay-trace: gateway did not become ready: %v\n", werr)
		return 2
	}
	srv.MarkReady()

	// --- The report header. ---
	fmt.Fprintf(out, "fak guard --replay-trace %s\n", fixturePath)
	fmt.Fprintf(out, "  wire   : %s (%s)\n", provider, guardtrace.InboundRoute(provider))
	fmt.Fprintf(out, "  floor  : %s\n", floorSource)
	fmt.Fprintf(out, "  trace  : %d turn(s)\n\n", len(f.Turns))

	const trace = "guard-replay"
	failures := 0
	var jseq0 uint64
	if j != nil {
		jseq0, _, _ = j.Stats()
	}

	client := &http.Client{Timeout: 120 * time.Second}
	for ti, turn := range f.Turns {
		raw, _, perr := guardtrace.PostTurn(client, gwURL, "X-Trace-Id", trace, provider, model, turn)
		if perr != nil {
			fmt.Fprintf(out, "turn %d: gateway error: %v\n", ti+1, perr)
			failures++
			continue
		}
		adjs, derr := guardtrace.DecodeAdjudications(raw)
		if derr != nil {
			fmt.Fprintf(out, "turn %d: decode error: %v\n", ti+1, derr)
			failures++
			continue
		}
		failures += renderReplayTurn(out, ti, turn, adjs)
	}

	if turns, _ := srv.VCacheTurnsSnapshot(); len(turns) > 0 {
		if snapPath, ok, werr := writeExplicitVCacheSnapshot(turns); werr != nil {
			fmt.Fprintf(out, "fak guard --replay-trace: vcache snapshot write failed: %v\n", werr)
			failures++
		} else if ok {
			fmt.Fprintf(out, "fak guard --replay-trace: wrote vcache snapshot - %d turn(s) to %s; replay with `FAK_VCACHE_SNAPSHOT=%s fak vcache score --json`.\n\n",
				len(turns), snapPath, snapPath)
		}
	}

	// --- The exit summary — the SAME roll-up the live `fak guard` prints on exit. ---
	fmt.Fprintln(out)
	fmt.Fprint(out, formatAuditSummary(srv.AdjudicationSummary()))
	if j != nil {
		if sum := formatJournalSummary(j, jseq0); sum != "" {
			fmt.Fprint(out, sum)
		}
		// Verify the chain at the path the ACTIVE journal actually writes to. journal.Enable
		// is idempotent on a process-global journal, so when a prior Enable already won (e.g.
		// an earlier replay in the same process) j is that journal — its Path(), not the temp
		// path this call computed, is the file on disk.
		if verifyPath := j.Path(); verifyPath != "" {
			if err := j.Flush(); err == nil {
				if n, verr := journal.Verify(verifyPath); verr != nil {
					fmt.Fprintf(out, "fak guard --replay-trace: journal chain FAILED to verify: %v\n", verr)
					failures++
				} else {
					fmt.Fprintf(out, "fak guard --replay-trace: journal chain verified — %d row(s), no tampering.\n", n)
				}
			}
		}
	}

	if failures > 0 {
		fmt.Fprintf(out, "\nfak guard --replay-trace: %d call(s) did NOT land on the fixture's expected disposition.\n", failures)
		return 1
	}
	fmt.Fprintln(out, "\nfak guard --replay-trace: every call landed on its expected disposition — the floor fired as recorded.")
	return 0
}

// renderReplayTurn prints one turn's per-call verdict lines and its token economy, and
// returns the number of calls whose verdict did NOT match the fixture's declared class.
func renderReplayTurn(out io.Writer, ti int, turn guardtrace.Turn, adjs []guardtrace.ResponseAdjudication) int {
	fmt.Fprintf(out, "turn %d:\n", ti+1)
	byTool := indexAdjudications(adjs)
	mismatches := 0
	for _, c := range turn.Calls {
		a, ok := takeAdjudication(byTool, c.Tool)
		got := "MISSING"
		mark := "✗"
		if ok {
			if a.Admitted {
				got = "ALLOW"
			} else {
				got = a.Kind
				if a.Reason != "" {
					got += "[" + a.Reason + "]"
				}
			}
		}
		want := "ALLOW"
		if !c.ExpectAllow() {
			want = "DENY"
			if c.Reason != "" {
				want += "[" + c.Reason + "]"
			}
		}
		matched := ok && a.Admitted == c.ExpectAllow() && (c.Reason == "" || a.Reason == c.Reason)
		if matched {
			mark = "✓"
		} else {
			mismatches++
		}
		preview := c.ArgPreview()
		if preview != "" {
			preview = " " + preview
		}
		fmt.Fprintf(out, "  %s %-6s %-8s%-50s -> %-22s (want %s)\n", mark, c.Tool, "", preview, got, want)
	}
	u := turn.Usage
	fmt.Fprintf(out, "  tokens : in=%d out=%d cache_read=%d cache_write=%d\n\n",
		u.InputTokens, u.OutputTokens, u.CacheReadInputTokens, u.CacheCreationInputTokens)
	return mismatches
}

// indexAdjudications groups the gateway's per-call verdicts by tool name so renderReplayTurn
// can pair each fixture call with its verdict positionally within a tool (the response only
// carries the tool name + verdict, not the fixture's call id).
func indexAdjudications(adjs []guardtrace.ResponseAdjudication) map[string][]guardtrace.ResponseAdjudication {
	m := map[string][]guardtrace.ResponseAdjudication{}
	for _, a := range adjs {
		m[a.Tool] = append(m[a.Tool], a)
	}
	return m
}

// takeAdjudication pops the next verdict for a tool (FIFO), so two calls to the same tool
// in one turn pair with their two verdicts in order.
func takeAdjudication(m map[string][]guardtrace.ResponseAdjudication, tool string) (guardtrace.ResponseAdjudication, bool) {
	q := m[tool]
	if len(q) == 0 {
		return guardtrace.ResponseAdjudication{}, false
	}
	a := q[0]
	m[tool] = q[1:]
	return a, true
}

// normalizeReplayWire maps the --replay-wire flag onto a provider, defaulting empty to the
// anthropic flagship and rejecting anything else.
func normalizeReplayWire(wire string) string {
	switch wire {
	case "", "anthropic", "claude":
		return "anthropic"
	case "openai", "codex", "opencode":
		return "openai"
	default:
		return ""
	}
}

func guardReplayAuditPlan(auditPath string, noAudit bool) (path string, optedOut bool, requested bool) {
	p := strings.TrimSpace(auditPath)
	if noAudit || strings.EqualFold(p, "off") {
		return "", true, false
	}
	if p != "" {
		return p, false, true
	}
	// The GENERATED default must be unique per PROCESS, not per instant (#5524). Deriving it
	// from time.Now().UnixNano() alone was not: the wall clock is backed by a coarse system
	// timer, so two back-to-back calls read the SAME nanosecond value (measured on Windows)
	// and two `fak guard replay` runs starting inside one tick resolved to one path. Nothing
	// then failed loudly — journal.Open uses O_CREATE|O_WRONLY|O_APPEND, so the loser
	// INTERLEAVED its hash-chained rows into the winner's file. The damage surfaced one layer
	// away as `journal: sequence gap: seq=1 want 3`, an integrity complaint against the
	// trace, while every per-call verdict still printed ✓ — pass-alone/fail-together, the
	// signature that gets misfiled as flake.
	//
	// guardSessionTempDir reserves a "fak-guard-replay-<pid>-<random>" dir via os.MkdirTemp,
	// which is ATOMIC (O_EXCL + retry) rather than merely improbable, and stamps the owning
	// PID so guardReapStaleTempDirs buries it once this process exits — strictly tidier than
	// the unbounded pile of loose per-run .jsonl files the clock-only name left behind.
	if dir, err := guardSessionTempDir("replay"); err == nil {
		return filepath.Join(dir, "audit.jsonl"), false, false
	}
	// Temp root not writable for a fresh dir: fall back to a PID-qualified name, so at least
	// concurrent processes cannot collide. Two plans inside ONE process can still repeat here,
	// which is sound because a replay asks for exactly one and journal.Enable is idempotent.
	return filepath.Join(os.TempDir(), fmt.Sprintf("fak-guard-replay-%d-%d.jsonl", os.Getpid(), time.Now().UnixNano())), false, false
}

func sameReplayAuditPath(a, b string) bool {
	if aa, err := filepath.Abs(a); err == nil {
		a = aa
	}
	if bb, err := filepath.Abs(b); err == nil {
		b = bb
	}
	return strings.EqualFold(filepath.Clean(a), filepath.Clean(b))
}

// replayUpstreamBase appends /v1 for the OpenAI wire (its adapter posts <base>/chat/completions)
// and leaves the Anthropic base bare (its adapter posts <base>/v1/messages) — the same split
// guardDefaultBaseURL / guardOpenAIV1Base apply on the live path.
func replayUpstreamBase(provider, host string) string {
	if provider == "openai" {
		return host + "/v1"
	}
	return host
}
