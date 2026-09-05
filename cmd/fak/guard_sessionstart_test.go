package main

import (
	"bytes"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/negframe"
	"github.com/anthony-chaudhary/fak/internal/procguard"
	"github.com/anthony-chaudhary/fak/internal/resume"
)

// stageGuardSessionStartWitness points the driver-pid witness (#5542) at a staged process table
// and a staged parent pid, so a test asserts on an ancestor chain it built rather than on the
// host's real process tree — and never pays for (or is flaked by) the real census.
func stageGuardSessionStartWitness(t *testing.T, parentPID int, procs []procguard.Proc) {
	t.Helper()
	prevProcs, prevPPID := guardSessionStartProcRelations, guardSessionStartParentPID
	t.Cleanup(func() { guardSessionStartProcRelations, guardSessionStartParentPID = prevProcs, prevPPID })
	guardSessionStartProcRelations = func() ([]procguard.Proc, string) { return procs, "" }
	guardSessionStartParentPID = func() int { return parentPID }
}

// TestGuardSessionStartEmitsAffordance asserts the #3092 affordance: the SessionStart hook
// emits a valid additionalContext envelope naming the fak entry verbs.
func TestGuardSessionStartEmitsAffordance(t *testing.T) {
	var out, errb bytes.Buffer
	code := runGuardSessionStart(&out, &errb, []string{"--mode", "on"})
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (SessionStart must never wedge a start)", code)
	}

	// Valid JSON with the exact Claude Code SessionStart envelope shape.
	var env struct {
		HookSpecificOutput struct {
			HookEventName     string `json:"hookEventName"`
			AdditionalContext string `json:"additionalContext"`
		} `json:"hookSpecificOutput"`
	}
	if err := json.Unmarshal(out.Bytes(), &env); err != nil {
		t.Fatalf("stdout is not valid JSON envelope: %v\n%s", err, out.String())
	}
	if env.HookSpecificOutput.HookEventName != "SessionStart" {
		t.Fatalf("hookEventName = %q, want SessionStart", env.HookSpecificOutput.HookEventName)
	}
	ctx := env.HookSpecificOutput.AdditionalContext
	for _, verb := range []string{"fak_capabilities", "fak_admit", "fak_tools_search"} {
		if !strings.Contains(ctx, verb) {
			t.Fatalf("affordance did not name entry verb %q: %s", verb, ctx)
		}
	}
}

// TestGuardSessionStartIdentity is the #4113 acceptance gate: a SessionStart hook holding both
// ids — the guard trace via --trace and the transcript UUID via CLAUDE_CODE_SESSION_ID — appends
// exactly one uuid<->trace join row to resume_identity.jsonl, and FoldIdentity (via LoadIdentity)
// resolves the UUID to the trace and back. This is the A1 store's first producer (#4112).
func TestGuardSessionStartIdentity(t *testing.T) {
	regDir := t.TempDir()
	t.Setenv("FLEET_REG_DIR", regDir)
	const uuid = "11111111-2222-3333-4444-555555555555"
	const trace = "trace-abc"
	t.Setenv("CLAUDE_CODE_SESSION_ID", uuid)
	// The hook runs as a child of the driver it is recording; stage that chain explicitly.
	stageGuardSessionStartWitness(t, 4242, []procguard.Proc{
		{PID: 4242, Name: "fak", PPID: procguard.IntPtr(9001), Cmdline: "fak guard-sessionstart --trace " + trace},
		{PID: 9001, Name: "claude", Cmdline: "claude -p do the work"},
	})

	var out, errb bytes.Buffer
	if code := runGuardSessionStart(&out, &errb, []string{"--mode", "on", "--trace", trace}); code != 0 {
		t.Fatalf("exit = %d, want 0 (SessionStart must never wedge a start)", code)
	}

	traceByUUID, uuidByTrace := resume.LoadIdentity(regDir)
	if got := traceByUUID[uuid]; got != trace {
		t.Fatalf("traceByUUID[%q] = %q, want %q", uuid, got, trace)
	}
	if got := uuidByTrace[trace]; got != uuid {
		t.Fatalf("uuidByTrace[%q] = %q, want %q", trace, got, uuid)
	}

	// Two rows on a fresh start where a driver WAS witnessed, and the order is the contract:
	// the uuid<->trace join lands first carrying no pid, the pid follows in a second append.
	// The witness reads the host process table, which costs about a second of wall clock on a
	// loaded host; folding it into the join row would put that second in front of the write and
	// a hook killed mid-census would lose a join it used to record (#4112). So the older
	// durability guarantee stays strictly ahead of the newer pid one, and this asserts the
	// ordering rather than the count — the count is just what the ordering implies.
	raw, err := os.ReadFile(resume.IdentityLedgerPath(regDir))
	if err != nil {
		t.Fatalf("read identity store: %v", err)
	}
	rows := resume.LoadIdentityRows(regDir)
	if len(rows) != 2 {
		t.Fatalf("want the join row then the pid row, got %d:\n%s", len(rows), raw)
	}
	if rows[0].PID != 0 {
		t.Fatalf("the FIRST row carries pid %d — the join must not wait behind the census\n%s", rows[0].PID, raw)
	}
	if rows[0].UUID != uuid || rows[0].Trace != trace {
		t.Fatalf("the first row is not a complete join: %+v", rows[0])
	}
	if rows[0].Provider != "claude" {
		t.Fatalf("the first row provider = %q, want claude: %+v", rows[0].Provider, rows[0])
	}
	if rows[1].PID != 9001 {
		t.Fatalf("the second row carries pid %d, want 9001\n%s", rows[1].PID, raw)
	}

	// #5542: the row also carries the DRIVER pid the hook witnessed. Without it a
	// first-generation `claude -p …` worker has no recorded pid anywhere on the host, so
	// `fak resume stopped` can never decide its liveness and defers it forever.
	if got := resume.FoldIdentityDriverPIDs(resume.LoadIdentityRows(regDir))[uuid]; got != 9001 {
		t.Fatalf("recorded driver pid = %d, want 9001 (the witnessed driver ancestor)\n%s", got, raw)
	}
}

// TestGuardSessionStartIdentityRecordsNoUnwitnessedPID pins the other half of the contract: a
// hook that cannot WITNESS which process it belongs to records no pid at all. Absence reads as
// "not recorded" downstream, which defers — a guess would read as a driver whose exit could
// later be mistaken for the death of the real one.
func TestGuardSessionStartIdentityRecordsNoUnwitnessedPID(t *testing.T) {
	const uuid = "22222222-3333-4444-5555-666666666666"
	for _, c := range []struct {
		name   string
		parent int
		procs  []procguard.Proc
	}{
		{
			name:   "no driver anywhere in the chain",
			parent: 4242,
			procs: []procguard.Proc{
				{PID: 4242, Name: "fak", PPID: procguard.IntPtr(700), Cmdline: "fak guard-sessionstart"},
				{PID: 700, Name: "sh", Cmdline: "/bin/sh -c make ci"},
			},
		},
		{
			// A wrapper that NAMES claude on its command line is not a claude process. Recording
			// it would bind the transcript to a program that outlives (or predeceases) the driver.
			name:   "a wrapper naming claude is not the driver",
			parent: 4242,
			procs: []procguard.Proc{
				{PID: 4242, Name: "fak", PPID: procguard.IntPtr(15696), Cmdline: "fak guard-sessionstart"},
				{PID: 15696, Name: "fak", Cmdline: "fak guard -- claude --dangerously-skip-permissions"},
			},
		},
		{
			name:   "an unobtainable census witnesses nothing",
			parent: 4242,
			procs:  nil,
		},
	} {
		t.Run(c.name, func(t *testing.T) {
			regDir := t.TempDir()
			t.Setenv("FLEET_REG_DIR", regDir)
			t.Setenv("CLAUDE_CODE_SESSION_ID", uuid)
			stageGuardSessionStartWitness(t, c.parent, c.procs)

			var out, errb bytes.Buffer
			if code := runGuardSessionStart(&out, &errb, []string{"--mode", "on", "--trace", "trace-abc"}); code != 0 {
				t.Fatalf("exit = %d, want 0 (SessionStart must never wedge a start)", code)
			}
			rows := resume.LoadIdentityRows(regDir)
			if len(rows) != 1 {
				t.Fatalf("want exactly one identity row, got %d: %+v", len(rows), rows)
			}
			if rows[0].PID != 0 {
				t.Fatalf("recorded pid = %d, want 0 (nothing was witnessed)", rows[0].PID)
			}
			// The join itself must still land: no witness costs the pid, never the row.
			if traceByUUID, _ := resume.LoadIdentity(regDir); traceByUUID[uuid] != "trace-abc" {
				t.Fatalf("the uuid<->trace join was lost when no pid could be witnessed: %+v", rows[0])
			}
		})
	}
}

// TestGuardSessionStartDriverPIDInWalk pins the walk itself. The hook is a child of the driver
// on a host that spawns hooks without a shell and a grandchild on one that does, so the rule is
// "nearest ancestor that IS the driver" — not a fixed depth, and never a command-line substring.
// Every shape it cannot witness returns 0 ("not recorded").
//
// Being the driver means either of two things, and the PAIR is the whole #5557 contract — a
// test of only the positive direction would be worthless here:
//
//   - the process IMAGE is the driver (#5542's rule, unchanged); or
//   - the image is `node` and its argv executes the driver's own ENTRYPOINT SCRIPT — the
//     node-wrapped install that #5542 could not witness and therefore deferred forever.
//
// The second arm is the only command line this witness reads, so the negative cases below are
// load-bearing: `fak guard -- claude …` (and its sharper twin, a `fak guard --` whose argv
// carries the entrypoint token verbatim) must still fail to match, or a resume could be fired
// against a wrapper's death while the driver it wraps is still writing the transcript.
func TestGuardSessionStartDriverPIDInWalk(t *testing.T) {
	for _, c := range []struct {
		name   string
		parent int
		procs  []procguard.Proc
		want   int
	}{
		{
			name:   "the direct parent is the driver",
			parent: 4242,
			procs: []procguard.Proc{
				{PID: 4242, Name: "fak", PPID: procguard.IntPtr(9001), Cmdline: "fak guard-sessionstart"},
				{PID: 9001, Name: "claude", PPID: procguard.IntPtr(15696), Cmdline: "claude -p do the work"},
				{PID: 15696, Name: "fak", Cmdline: "fak guard -- claude"},
			},
			want: 9001,
		},
		{
			name:   "a shell between the hook and the driver is walked through",
			parent: 4242,
			procs: []procguard.Proc{
				{PID: 4242, Name: "fak", PPID: procguard.IntPtr(700), Cmdline: "fak guard-sessionstart"},
				{PID: 700, Name: "sh", PPID: procguard.IntPtr(9001), Cmdline: "/bin/sh -c fak guard-sessionstart"},
				{PID: 9001, Name: "claude", Cmdline: "claude -p do the work"},
			},
			want: 9001,
		},
		{
			name:   "the NEAREST driver ancestor wins over an outer one",
			parent: 4242,
			procs: []procguard.Proc{
				{PID: 4242, Name: "fak", PPID: procguard.IntPtr(9001), Cmdline: "fak guard-sessionstart"},
				{PID: 9001, Name: "claude", PPID: procguard.IntPtr(9002), Cmdline: "claude -p inner"},
				{PID: 9002, Name: "claude", Cmdline: "claude -p outer"},
			},
			want: 9001,
		},
		{
			// #5557, half one of the contract: a driver launched through a Node wrapper presents
			// the image `node`, so the image rule alone never named it and the walk ran out of
			// hops — a permanent defer for that whole install shape. The argv test admits it by
			// its ENTRYPOINT SCRIPT (the npm package's cli.js), not by the word "claude".
			name:   "a node-wrapped driver is witnessed through its entrypoint script",
			parent: 4242,
			procs: []procguard.Proc{
				{PID: 4242, Name: "fak", PPID: procguard.IntPtr(9001), Cmdline: "fak guard-sessionstart"},
				{PID: 9001, Name: "node", Cmdline: "/usr/bin/node /home/worker/.npm-global/lib/node_modules/@anthropic-ai/claude-code/cli.js -p do the work"},
			},
			want: 9001,
		},
		{
			// The same shape as Windows spells it: a `node.exe` image and a backslash argv.
			name:   "a windows node-wrapped driver is witnessed through its entrypoint script",
			parent: 4242,
			procs: []procguard.Proc{
				{PID: 4242, Name: "fak.exe", PPID: procguard.IntPtr(9001), Cmdline: "fak guard-sessionstart"},
				// Deliberately rooted off a drive rather than a user home: an operator-home path
				// shape in a fixture trips the SECRET_SHAPE commit gate, and nothing here needs
				// one — what the case pins is the backslash argv and the npm package layout.
				{PID: 9001, Name: `D:\tools\nodejs\node.exe`, Cmdline: `"D:\tools\nodejs\node.exe" D:\appdata\npm\node_modules\@anthropic-ai\claude-code\cli.js -p do the work`},
			},
			want: 9001,
		},
		{
			// The local-install shape (`claude/cli.js`), which is also the driver command line
			// the resume-liveness fixture in this package already models.
			name:   "a node-wrapped driver in a claude dir is witnessed",
			parent: 4242,
			procs: []procguard.Proc{
				{PID: 4242, Name: "fak", PPID: procguard.IntPtr(9001), Cmdline: "fak guard-sessionstart"},
				{PID: 9001, Name: "node", Cmdline: "node /opt/claude/cli.js -p do the work"},
			},
			want: 9001,
		},
		{
			// #5557, half two — and the half that makes half one safe. The `fak guard --` wrapper
			// sits one level ABOVE the driver on this host and names it on argv; recording it
			// would bind the transcript to a program that outlives (or predeceases) the driver.
			// Its IMAGE is `fak`, so it never reaches the argv test at all.
			name:   "a fak guard wrapper naming claude on argv is not the driver",
			parent: 4242,
			procs: []procguard.Proc{
				{PID: 4242, Name: "fak", PPID: procguard.IntPtr(15696), Cmdline: "fak guard-sessionstart"},
				{PID: 15696, Name: "fak", Cmdline: "fak guard -- claude --dangerously-skip-permissions -p do the work"},
			},
			want: 0,
		},
		{
			// The sharpest form of the same negative: the wrapper's argv carries the ENTRYPOINT
			// token verbatim, so only the `node` image gate can reject it. This is the case a
			// cmdline-only rule would wrongly admit.
			name:   "a fak guard wrapper naming the node entrypoint is not the driver",
			parent: 4242,
			procs: []procguard.Proc{
				{PID: 4242, Name: "fak", PPID: procguard.IntPtr(15696), Cmdline: "fak guard-sessionstart"},
				{PID: 15696, Name: "fak", Cmdline: "fak guard -- node /home/worker/.npm-global/lib/node_modules/@anthropic-ai/claude-code/cli.js -p do the work"},
			},
			want: 0,
		},
		{
			// A node process that merely MENTIONS a path under a claude config dir is some other
			// program (an MCP server, a hook). It executes no driver entrypoint, so it is not it.
			name:   "a node process merely naming a claude path is not the driver",
			parent: 4242,
			procs: []procguard.Proc{
				{PID: 4242, Name: "fak", PPID: procguard.IntPtr(9001), Cmdline: "fak guard-sessionstart"},
				{PID: 9001, Name: "node", Cmdline: "node /srv/tools/mcp-server.js --settings /home/worker/.claude/settings.json"},
			},
			want: 0,
		},
		{
			// The entrypoint is compared as a (dir, base) TOKEN pair, so a directory that merely
			// ends in "claude" is a different token — a substring rule would admit this.
			name:   "a node entrypoint under a lookalike directory is not the driver",
			parent: 4242,
			procs: []procguard.Proc{
				{PID: 4242, Name: "fak", PPID: procguard.IntPtr(9001), Cmdline: "fak guard-sessionstart"},
				{PID: 9001, Name: "node", Cmdline: "node /srv/notclaude/cli.js --serve"},
			},
			want: 0,
		},
		{
			name:   "a chain that leaves the snapshot witnesses nothing",
			parent: 4242,
			procs: []procguard.Proc{
				{PID: 4242, Name: "fak", PPID: procguard.IntPtr(9001), Cmdline: "fak guard-sessionstart"},
			},
			want: 0,
		},
		{
			name:   "a row with no parent ends the walk",
			parent: 4242,
			procs:  []procguard.Proc{{PID: 4242, Name: "fak", Cmdline: "fak guard-sessionstart"}},
			want:   0,
		},
		{
			// A recycled ppid can close a loop; the walk must terminate rather than spin.
			name:   "a ppid cycle terminates without a witness",
			parent: 1,
			procs: []procguard.Proc{
				{PID: 1, Name: "a", PPID: procguard.IntPtr(2)},
				{PID: 2, Name: "b", PPID: procguard.IntPtr(1)},
			},
			want: 0,
		},
		{
			name:   "a zero parent pid witnesses nothing",
			parent: 0,
			procs:  []procguard.Proc{{PID: 9001, Name: "claude", Cmdline: "claude -p do the work"}},
			want:   0,
		},
	} {
		t.Run(c.name, func(t *testing.T) {
			if got := guardSessionStartDriverPIDIn(c.parent, c.procs); got != c.want {
				t.Fatalf("guardSessionStartDriverPIDIn(%d) = %d, want %d", c.parent, got, c.want)
			}
		})
	}
}

// TestGuardSessionStartDriverPIDInHopBound pins the bound: a chain longer than the hop budget
// stops rather than walking off into whatever a recycled ppid points at.
func TestGuardSessionStartDriverPIDInHopBound(t *testing.T) {
	var procs []procguard.Proc
	// pids 1..N, each the child of the next; the driver sits one hop PAST the bound.
	driver := guardSessionStartAncestorHops + 1
	for pid := 1; pid <= driver; pid++ {
		p := procguard.Proc{PID: pid, Name: "fak", Cmdline: "fak guard-sessionstart"}
		if pid < driver {
			p.PPID = procguard.IntPtr(pid + 1)
		} else {
			p.Name, p.Cmdline = "claude", "claude -p do the work"
		}
		procs = append(procs, p)
	}
	if got := guardSessionStartDriverPIDIn(1, procs); got != 0 {
		t.Fatalf("a driver beyond the hop bound was recorded as %d, want 0 (not witnessed)", got)
	}
	// One hop closer and it IS reachable — the bound is what stopped the walk, not the rule.
	if got := guardSessionStartDriverPIDIn(2, procs); got != driver {
		t.Fatalf("driver within the hop bound = %d, want %d", got, driver)
	}
}

// TestGuardSessionStartIdentityFailOpen asserts the hook's fail-open contract: a resumed child
// (CLAUDE_CODE_SESSION_ID stripped, so the UUID is blank) writes NO half row — a join needs both
// endpoints — yet still exits 0.
func TestGuardSessionStartIdentityFailOpen(t *testing.T) {
	regDir := t.TempDir()
	t.Setenv("FLEET_REG_DIR", regDir)
	t.Setenv("CLAUDE_CODE_SESSION_ID", "") // resumed child: transcript UUID absent

	var out, errb bytes.Buffer
	if code := runGuardSessionStart(&out, &errb, []string{"--mode", "on", "--trace", "trace-abc"}); code != 0 {
		t.Fatalf("exit = %d, want 0", code)
	}
	if _, err := os.Stat(resume.IdentityLedgerPath(regDir)); !os.IsNotExist(err) {
		t.Fatalf("a half row (missing UUID) must not be written; stat err = %v", err)
	}
}

// TestGuardSessionStartArgsThreadsTrace pins the install-side half: a non-empty trace threads
// --trace into the hook argv (so the running hook holds it), an empty one threads nothing.
func TestGuardSessionStartArgsThreadsTrace(t *testing.T) {
	joined := strings.Join(guardSessionStartArgs(true, "trace-xyz"), " ")
	if !strings.Contains(joined, "--trace trace-xyz") {
		t.Fatalf("args missing threaded --trace: %q", joined)
	}
	if !strings.Contains(joined, "--managed") {
		t.Fatalf("args dropped --managed: %q", joined)
	}
	if strings.Contains(strings.Join(guardSessionStartArgs(false, ""), " "), "--trace") {
		t.Fatalf("an empty trace must not emit a --trace flag")
	}
}

// TestInstallGuardSessionStartHookThreadsTrace asserts the trace threads all the way through
// install into the written SessionStart hook settings the child actually loads.
func TestInstallGuardSessionStartHookThreadsTrace(t *testing.T) {
	dir := t.TempDir()
	_, install, err := installGuardSessionStartHookAt([]string{"claude", "-p", "x"}, "on", true, "fak", dir, "", "trace-xyz")
	if err != nil {
		t.Fatalf("install: %v", err)
	}
	raw, err := os.ReadFile(install.SettingsPath)
	if err != nil {
		t.Fatalf("read written settings: %v", err)
	}
	if !strings.Contains(string(raw), "trace-xyz") {
		t.Fatalf("written SessionStart hook did not thread the trace id: %s", raw)
	}
}

// TestGuardSessionStartManagedInjectsRule asserts the spine (#3512): a --managed (headless)
// SessionStart injection carries the long-horizon persistence + managed-context rule ON TOP of
// the base affordance, while a plain (attended) injection carries the affordance alone.
func TestGuardSessionStartManagedInjectsRule(t *testing.T) {
	readCtx := func(argv []string) string {
		var out, errb bytes.Buffer
		if code := runGuardSessionStart(&out, &errb, argv); code != 0 {
			t.Fatalf("exit = %d for %v", code, argv)
		}
		var env struct {
			HookSpecificOutput struct {
				AdditionalContext string `json:"additionalContext"`
			} `json:"hookSpecificOutput"`
		}
		if err := json.Unmarshal(out.Bytes(), &env); err != nil {
			t.Fatalf("not valid JSON for %v: %v", argv, err)
		}
		return env.HookSpecificOutput.AdditionalContext
	}

	managed := readCtx([]string{"--mode", "on", "--managed"})
	for _, want := range []string{"fak_capabilities", "managed context is ON", "CHECKPOINT", "REBUILD", "TOOL_WIDTH_HINT", "independent", "dependent calls sequential"} {
		if !strings.Contains(managed, want) {
			t.Fatalf("managed injection missing %q: %s", want, managed)
		}
	}

	plain := readCtx([]string{"--mode", "on"})
	if !strings.Contains(plain, "fak_capabilities") {
		t.Fatalf("plain injection dropped the base affordance: %s", plain)
	}
	if strings.Contains(plain, "managed context is ON") {
		t.Fatalf("plain (attended) injection must NOT carry the long-horizon rule: %s", plain)
	}
}

// TestSessionStartRulePositiveVoice pins the #3566 emit-time guarantee at the SessionStart
// boundary: the additionalContext fak injects is already in positive-voice NORMAL FORM — routing
// it through the deterministic reframe again is a no-op (fixed point) — while every load-bearing
// token (the MCP entry verbs and the managed-context directives) survives the pass byte-for-byte.
func TestSessionStartRulePositiveVoice(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runGuardSessionStart(&out, &errb, []string{"--mode", "on", "--managed"}); code != 0 {
		t.Fatalf("exit = %d", code)
	}
	var env struct {
		HookSpecificOutput struct {
			AdditionalContext string `json:"additionalContext"`
		} `json:"hookSpecificOutput"`
	}
	if err := json.Unmarshal(out.Bytes(), &env); err != nil {
		t.Fatalf("not valid JSON: %v", err)
	}
	ctx := env.HookSpecificOutput.AdditionalContext
	// Fixed point: what fak injects has already been through the reframe, so a second pass changes
	// nothing. This is the always-on invariant — no injected string reaches the model un-normalized.
	if reframed := negframe.Reframe(ctx); reframed != ctx {
		t.Fatalf("injected context is not a positive-voice fixed point:\n have: %q\n want: %q", ctx, reframed)
	}
	// The reframe preserved the load-bearing structure it must never mangle.
	for _, tok := range []string{"fak_capabilities", "managed context is ON", "CHECKPOINT", "REBUILD"} {
		if !strings.Contains(ctx, tok) {
			t.Fatalf("reframe dropped load-bearing token %q:\n%s", tok, ctx)
		}
	}
}

// TestGuardSessionStartManagedDefault locks the default-on admission (#3512): a headless
// `claude -p` fleet worker is admitted MANAGED (gets the long-horizon rule) by default, while
// an attended interactive `claude` TUI is not. This is the switch that makes the persistence
// posture on-by-default exactly where a human is not present to keep the session going.
func TestGuardSessionStartManagedDefault(t *testing.T) {
	if !guardSessionStartManaged([]string{"claude", "-p", "do the work"}) {
		t.Fatalf("a headless `claude -p` worker must default to MANAGED")
	}
	if guardSessionStartManaged([]string{"claude"}) {
		t.Fatalf("an attended `claude` TUI must not be forced onto the managed posture")
	}
}

// TestGuardSessionStartOffSuppresses asserts the off knob emits nothing (a lean harness
// opts out).
func TestGuardSessionStartOffSuppresses(t *testing.T) {
	var out, errb bytes.Buffer
	code := runGuardSessionStart(&out, &errb, []string{"--mode", "off"})
	if code != 0 {
		t.Fatalf("exit = %d, want 0", code)
	}
	if out.Len() != 0 {
		t.Fatalf("off mode should emit nothing, got: %q", out.String())
	}
}

// TestGuardSessionStartDefaultsOn asserts an empty mode defaults to on (the affordance is
// the fix, so it is on by default).
func TestGuardSessionStartDefaultsOn(t *testing.T) {
	if got := normalizeGuardSessionStartMode(""); got != guardSessionStartModeOn {
		t.Fatalf("empty mode = %q, want on", got)
	}
	if got := normalizeGuardSessionStartMode("OFF"); got != guardSessionStartModeOff {
		t.Fatalf("OFF (case-insensitive) = %q, want off", got)
	}
}

// TestGuardSessionStartSettingsRoundTrip asserts the settings writer emits a SessionStart
// hook entry the merge path can read back, and that the merge preserves a sibling hook.
func TestGuardSessionStartSettingsRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/settings.json"

	// Seed a settings file that already carries a Stop hook (a sibling the merge must keep).
	seed := guardPreCompactClaudeSettings{Hooks: map[string][]guardPreCompactClaudeMatcher{
		"Stop": guardStopHookMatchers("fak"),
	}}
	seedData, _ := json.MarshalIndent(seed, "", "  ")
	if err := os.WriteFile(path, seedData, 0o600); err != nil {
		t.Fatalf("seed: %v", err)
	}

	if err := mergeGuardSessionStartIntoSettings(path, "fak", false, ""); err != nil {
		t.Fatalf("merge: %v", err)
	}

	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read back: %v", err)
	}
	var got guardPreCompactClaudeSettings
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("parse merged: %v", err)
	}
	if _, ok := got.Hooks["SessionStart"]; !ok {
		t.Fatalf("merged settings missing SessionStart hook: %s", raw)
	}
	if _, ok := got.Hooks["Stop"]; !ok {
		t.Fatalf("merge dropped the sibling Stop hook: %s", raw)
	}
	// The SessionStart hook must invoke the guard-sessionstart verb.
	if !strings.Contains(string(raw), "guard-sessionstart") {
		t.Fatalf("SessionStart hook does not invoke guard-sessionstart: %s", raw)
	}
}

// TestInstallGuardSessionStartHookAtWiring covers the install path cmdGuard actually invokes
// (guard.go -> installGuardSessionStartHook -> installGuardSessionStartHookAt) for the #3092
// affordance. The actuator/merge tests above exercise emission and merge in isolation; this
// asserts the install BRANCHING that reaches the child — a claude launcher gets the SessionStart
// hook wired into its --settings, a non-claude child and the off knob stay no-ops, and merging
// into an existing guard settings file preserves the sibling hooks. Without this, a regression
// that stops wiring the affordance (re-inerting the fak verbs) would pass every existing test.
func TestInstallGuardSessionStartHookAtWiring(t *testing.T) {
	t.Run("claude child gets a fresh settings file and a --settings repoint", func(t *testing.T) {
		dir := t.TempDir()
		cmd := []string{"claude", "-p", "do the work"}
		out, install, err := installGuardSessionStartHookAt(cmd, "on", true, "fak", dir, "", "")
		if err != nil {
			t.Fatalf("install: %v", err)
		}
		if !install.Applied {
			t.Fatalf("expected Applied=true for a claude child, got %+v", install)
		}
		if !install.Managed {
			t.Fatalf("expected Managed=true for a headless claude -p child, got %+v", install)
		}
		if install.SettingsPath == "" {
			t.Fatalf("expected a written settings path, got empty")
		}
		// The launcher must stay first and gain `--settings <path>` so Claude Code loads the hook.
		if len(out) == 0 || out[0] != "claude" {
			t.Fatalf("launcher token moved or dropped: %v", out)
		}
		if !strings.Contains(strings.Join(out, " "), "--settings "+install.SettingsPath) {
			t.Fatalf("command missing --settings %s: %v", install.SettingsPath, out)
		}
		raw, err := os.ReadFile(install.SettingsPath)
		if err != nil {
			t.Fatalf("read written settings: %v", err)
		}
		var got guardPreCompactClaudeSettings
		if err := json.Unmarshal(raw, &got); err != nil {
			t.Fatalf("parse written settings: %v", err)
		}
		if _, ok := got.Hooks["SessionStart"]; !ok {
			t.Fatalf("written settings missing SessionStart hook: %s", raw)
		}
		if !strings.Contains(string(raw), "guard-sessionstart") {
			t.Fatalf("SessionStart hook does not invoke guard-sessionstart: %s", raw)
		}
		// A managed (headless) install must carry --managed so the injected context includes
		// the long-horizon persistence + managed-context rule (#3512).
		if !strings.Contains(string(raw), "--managed") {
			t.Fatalf("managed SessionStart hook missing --managed arg: %s", raw)
		}
	})

	t.Run("codex child gets a per-launch SessionStart hook", func(t *testing.T) {
		dir := t.TempDir()
		cmd := []string{"codex"}
		out, install, err := installGuardSessionStartHookAt(cmd, "on", false, "fak", dir, "", "trace-codex")
		if err != nil {
			t.Fatalf("install: %v", err)
		}
		if !install.Applied {
			t.Fatalf("expected Applied=true for a codex child, got %+v", install)
		}
		joined := strings.Join(out, " ")
		if !strings.Contains(joined, "hooks.SessionStart") || !strings.Contains(joined, "guard-sessionstart") {
			t.Fatalf("codex command missing per-launch SessionStart adapter: %v", out)
		}
	})

	t.Run("non-claude child is a no-op", func(t *testing.T) {
		cmd := []string{"bash", "-c", "echo hi"}
		out, install, err := installGuardSessionStartHookAt(cmd, "on", true, "fak", t.TempDir(), "", "")
		if err != nil {
			t.Fatalf("install: %v", err)
		}
		if install.Applied {
			t.Fatalf("expected no-op for a non-claude child, got %+v", install)
		}
		if install.Reason != "non-claude-child" {
			t.Fatalf("reason = %q, want non-claude-child", install.Reason)
		}
		if strings.Join(out, " ") != strings.Join(cmd, " ") {
			t.Fatalf("non-claude command was mutated: %v", out)
		}
	})

	t.Run("off mode stays a no-op even for a claude child", func(t *testing.T) {
		cmd := []string{"claude", "-p", "x"}
		out, install, err := installGuardSessionStartHookAt(cmd, "off", true, "fak", t.TempDir(), "", "")
		if err != nil {
			t.Fatalf("install: %v", err)
		}
		if install.Applied {
			t.Fatalf("off mode should not apply, got %+v", install)
		}
		if install.Reason != "disabled" {
			t.Fatalf("reason = %q, want disabled", install.Reason)
		}
		if strings.Join(out, " ") != strings.Join(cmd, " ") {
			t.Fatalf("off mode mutated the command: %v", out)
		}
	})

	t.Run("merges into an existing settings file without re-pointing the command", func(t *testing.T) {
		dir := t.TempDir()
		existing := dir + "/settings.json"
		seed := guardPreCompactClaudeSettings{Hooks: map[string][]guardPreCompactClaudeMatcher{
			"Stop": guardStopHookMatchers("fak"),
		}}
		seedData, _ := json.MarshalIndent(seed, "", "  ")
		if err := os.WriteFile(existing, seedData, 0o600); err != nil {
			t.Fatalf("seed: %v", err)
		}
		cmd := []string{"claude", "-p", "x"}
		out, install, err := installGuardSessionStartHookAt(cmd, "on", true, "fak", "", existing, "")
		if err != nil {
			t.Fatalf("install: %v", err)
		}
		if !install.Applied || install.SettingsPath != existing {
			t.Fatalf("expected merge into %s, got %+v", existing, install)
		}
		// The merge branch reuses the existing --settings file, so the command is not repointed.
		if strings.Join(out, " ") != strings.Join(cmd, " ") {
			t.Fatalf("merge branch should not append --settings again: %v", out)
		}
		raw, err := os.ReadFile(existing)
		if err != nil {
			t.Fatalf("read merged settings: %v", err)
		}
		var got guardPreCompactClaudeSettings
		if err := json.Unmarshal(raw, &got); err != nil {
			t.Fatalf("parse merged settings: %v", err)
		}
		if _, ok := got.Hooks["SessionStart"]; !ok {
			t.Fatalf("merge missing SessionStart hook: %s", raw)
		}
		if _, ok := got.Hooks["Stop"]; !ok {
			t.Fatalf("merge dropped the sibling Stop hook: %s", raw)
		}
	})
}

func TestGuardSessionStartHintPositiveFirst(t *testing.T) {
	if !strings.HasPrefix(guardSessionStartHint, "Reach for the fak substrate verbs") {
		t.Fatalf("hint does not lead with affordance: %q", guardSessionStartHint)
	}
	for _, forbidden := range []string{"before working as", "must invoke", "will not", "do not", "never"} {
		if strings.Contains(strings.ToLower(guardSessionStartHint), forbidden) {
			t.Fatalf("hint retains negation-first clause %q: %q", forbidden, guardSessionStartHint)
		}
	}
	for _, token := range []string{"`mcp__fak__fak_capabilities`", "`mcp__fak__fak_admit`", "`mcp__fak__fak_adjudicate`", "`mcp__fak__fak_memory_run`", "`mcp__fak__fak_tools_search`"} {
		if !strings.Contains(guardSessionStartHint, token) {
			t.Fatalf("hint dropped %s", token)
		}
	}
	if got := negframe.Reframe(guardSessionStartHint); got != guardSessionStartHint {
		t.Fatalf("positive source is not reframe-idempotent:\n got %q\nwant %q", got, guardSessionStartHint)
	}
}

func TestGuardSessionStartHintCodexPositiveFirst(t *testing.T) {
	hint := guardSessionStartHintForProvider("codex")
	if !strings.HasPrefix(hint, "Reach for the fak substrate verbs") {
		t.Fatalf("hint does not lead with affordance: %q", hint)
	}
	for _, forbidden := range []string{"before working as", "must invoke", "will not", "do not", "never"} {
		if strings.Contains(strings.ToLower(hint), forbidden) {
			t.Fatalf("hint retains negation-first clause %q: %q", forbidden, hint)
		}
	}
	if !strings.Contains(hint, "(MCP server `fak_guard`)") {
		t.Fatalf("hint dropped MCP server `fak_guard`: %s", hint)
	}
	if strings.Contains(hint, "(MCP server `fak`)") {
		t.Fatalf("hint retains (MCP server `fak`): %s", hint)
	}
	for _, token := range []string{
		"`mcp__fak_guard__fak_capabilities`",
		"`mcp__fak_guard__fak_admit`",
		"`mcp__fak_guard__fak_adjudicate`",
		"`mcp__fak_guard__fak_memory_run`",
		"`mcp__fak_guard__fak_tools_search`",
	} {
		if !strings.Contains(hint, token) {
			t.Fatalf("hint dropped %s", token)
		}
	}
	if strings.Contains(hint, "mcp__fak__") {
		t.Fatalf("hint retains mcp__fak__: %q", hint)
	}
	if got := negframe.Reframe(hint); got != hint {
		t.Fatalf("positive source is not reframe-idempotent:\n got %q\nwant %q", got, hint)
	}
}

func TestGuardSessionStartCodexProviderAffordance(t *testing.T) {
	var out, errb bytes.Buffer
	code := runGuardSessionStart(&out, &errb, []string{"--mode", "on", "--provider", "codex"})
	if code != 0 {
		t.Fatalf("exit = %d, want 0", code)
	}

	var env struct {
		HookSpecificOutput struct {
			HookEventName     string `json:"hookEventName"`
			AdditionalContext string `json:"additionalContext"`
		} `json:"hookSpecificOutput"`
	}
	if err := json.Unmarshal(out.Bytes(), &env); err != nil {
		t.Fatalf("stdout is not valid JSON envelope: %v\n%s", err, out.String())
	}
	if env.HookSpecificOutput.HookEventName != "SessionStart" {
		t.Fatalf("hookEventName = %q, want SessionStart", env.HookSpecificOutput.HookEventName)
	}
	ctx := env.HookSpecificOutput.AdditionalContext

	if !strings.Contains(ctx, "(MCP server `fak_guard`)") {
		t.Fatalf("affordance missing '(MCP server `fak_guard`)': %s", ctx)
	}
	if strings.Contains(ctx, "(MCP server `fak`)") {
		t.Fatalf("affordance retains '(MCP server `fak`)' for codex: %s", ctx)
	}
	for _, verb := range []string{
		"mcp__fak_guard__fak_capabilities",
		"mcp__fak_guard__fak_admit",
		"mcp__fak_guard__fak_adjudicate",
		"mcp__fak_guard__fak_memory_run",
		"mcp__fak_guard__fak_tools_search",
	} {
		if !strings.Contains(ctx, verb) {
			t.Fatalf("affordance did not name entry verb %q: %s", verb, ctx)
		}
	}
	if strings.Contains(ctx, "mcp__fak__") {
		t.Fatalf("affordance still contains mcp__fak__ for codex: %s", ctx)
	}

	if !strings.HasPrefix(ctx, "Reach for the fak substrate verbs") {
		t.Fatalf("codex hint does not lead with affordance: %q", ctx)
	}
	for _, forbidden := range []string{"before working as", "must invoke", "will not", "do not", "never"} {
		if strings.Contains(strings.ToLower(ctx), forbidden) {
			t.Fatalf("codex hint retains negation clause %q: %q", forbidden, ctx)
		}
	}
	if reframed := negframe.Reframe(ctx); reframed != ctx {
		t.Fatalf("codex hint is not a positive-voice fixed point:\n have: %q\n want: %q", ctx, reframed)
	}
}

func TestGuardSessionStartCodexProviderManagedAffordance(t *testing.T) {
	var out, errb bytes.Buffer
	code := runGuardSessionStart(&out, &errb, []string{"--mode", "on", "--provider", "codex", "--managed"})
	if code != 0 {
		t.Fatalf("exit = %d, want 0", code)
	}

	var env struct {
		HookSpecificOutput struct {
			AdditionalContext string `json:"additionalContext"`
		} `json:"hookSpecificOutput"`
	}
	if err := json.Unmarshal(out.Bytes(), &env); err != nil {
		t.Fatalf("stdout is not valid JSON envelope: %v\n%s", err, out.String())
	}
	ctx := env.HookSpecificOutput.AdditionalContext

	if !strings.Contains(ctx, "(MCP server `fak_guard`)") {
		t.Fatalf("managed codex hint missing '(MCP server `fak_guard`)': %s", ctx)
	}
	for _, verb := range []string{
		"mcp__fak_guard__fak_context_value",
		"mcp__fak_guard__fak_context_spans",
		"mcp__fak_guard__fak_context_restore",
	} {
		if !strings.Contains(ctx, verb) {
			t.Fatalf("managed codex hint did not name session-state tool %q: %s", verb, ctx)
		}
	}
	if strings.Contains(ctx, "mcp__fak__") {
		t.Fatalf("managed codex hint still contains mcp__fak__: %s", ctx)
	}
	if reframed := negframe.Reframe(ctx); reframed != ctx {
		t.Fatalf("managed codex hint is not a positive-voice fixed point:\n have: %q\n want: %q", ctx, reframed)
	}
}

// TestGuardSessionStartWritesNegframeJournal is #5365's witness for the halves #3568 deferred.
// Before this, the read side (guardNegframeSummaryLine) was structurally silent because NOTHING
// wrote guardNegframeJournalRel, and the emit called negframe.ReframeFakOnly unconditionally so
// #3546's control arm shipped reframed prose anyway. Both are asserted end-to-end here, through
// the real hook actuator rather than the helper in isolation:
//
//   - a SessionStart emit leaves a foldable row, so the exit summary has something to report;
//   - the row names the arm the FAK_ABLATE_NEGFRAME_REFRAME lever actually selected;
//   - on the control arm the injected prose is byte-identical to the raw fragment.
func TestGuardSessionStartWritesNegframeJournal(t *testing.T) {
	// guardNegframeJournalRel is workspace-relative, so each emit runs inside a scratch tree —
	// that also keeps the assertion off this repo's real .fak/ journal.
	emit := func(t *testing.T, argv ...string) (ctx, summary string) {
		t.Helper()
		t.Chdir(t.TempDir())
		var out, errb bytes.Buffer
		if code := runGuardSessionStart(&out, &errb, append([]string{"--mode", "on"}, argv...)); code != 0 {
			t.Fatalf("exit = %d, want 0 (SessionStart must never wedge a start)", code)
		}
		var env struct {
			HookSpecificOutput struct {
				AdditionalContext string `json:"additionalContext"`
			} `json:"hookSpecificOutput"`
		}
		if err := json.Unmarshal(out.Bytes(), &env); err != nil {
			t.Fatalf("not valid JSON: %v\n%s", err, out.String())
		}
		return env.HookSpecificOutput.AdditionalContext, guardNegframeSummaryLine(guardNegframeJournalRel)
	}

	t.Run("treatment arm leaves a foldable row", func(t *testing.T) {
		t.Setenv("FAK_ABLATE", "")
		t.Setenv(guardNegframeEnvVar, "1")
		ctx, summary := emit(t, "--managed")
		if summary == "" {
			t.Fatal("SessionStart wrote no journal row — the exit-summary fold is still silent (#5365 item 2)")
		}
		if !strings.Contains(summary, "reframe on") {
			t.Fatalf("row did not record the treatment arm:\n%s", summary)
		}
		if !strings.Contains(ctx, "fak_capabilities") {
			t.Fatalf("routing through the lever dropped the affordance:\n%s", ctx)
		}
	})

	// The control arm is the whole point of #3546: with the lever off, the bytes fak injects are
	// the RAW fragment, not a quietly-reframed one, and the row says so.
	t.Run("control arm records the ablated arm and ships raw bytes", func(t *testing.T) {
		t.Setenv("FAK_ABLATE", "")
		t.Setenv(guardNegframeEnvVar, "0")
		ctx, summary := emit(t)
		if summary == "" {
			t.Fatal("control arm wrote no journal row")
		}
		if !strings.Contains(summary, "reframe OFF") {
			t.Fatalf("row did not record the ablated arm:\n%s", summary)
		}
		if ctx != guardSessionStartHint {
			t.Fatalf("control arm rewrote the injected prose:\n got %q\nwant %q", ctx, guardSessionStartHint)
		}
	})

	// Begin, not append: a second SessionStart is a new session boundary, so the fold reports
	// THIS session rather than accumulating the workspace's whole history.
	t.Run("each session start resets the fold", func(t *testing.T) {
		t.Setenv("FAK_ABLATE", "")
		t.Setenv(guardNegframeEnvVar, "1")
		dir := t.TempDir()
		t.Chdir(dir)
		var out, errb bytes.Buffer
		for range 3 {
			if code := runGuardSessionStart(&out, &errb, []string{"--mode", "on"}); code != 0 {
				t.Fatalf("exit = %d, want 0", code)
			}
		}
		raw, err := os.ReadFile(guardNegframeJournalRel)
		if err != nil {
			t.Fatalf("read journal: %v", err)
		}
		if n := len(strings.Split(strings.TrimSpace(string(raw)), "\n")); n != 1 {
			t.Fatalf("three session starts left %d rows, want 1 (the boundary must truncate):\n%s", n, raw)
		}
	})
}

func TestSessionStartBatchingPostureCanStayShadowDisabled(t *testing.T) {
	t.Setenv("FAK_TOOL_WIDTH_HINT", "off")
	var out, errb bytes.Buffer
	if code := runGuardSessionStart(&out, &errb, []string{"--mode", "on", "--managed"}); code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	if strings.Contains(out.String(), "TOOL_WIDTH_HINT") {
		t.Fatalf("disabled posture injected: %s", out.String())
	}
}
