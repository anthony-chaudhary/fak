package procguard

// collect_posix_test.go — the witnesses for #5385, where every POSIX census came back
// empty on macOS. Two independent defects had to line up for that: `ps` was asked for
// procps-only keywords BSD `ps` does not have (it printed the whole table anyway and
// exited 1), and runTool threw the printed table away because the exit code was non-zero.
//
// Every pre-existing test in this package injects synthetic []Proc and starts AFTER the
// collectors, which is exactly why macOS CI (build + vet only) could stay green while the
// guard was inert on every Mac. So the tests below deliberately aim at the seam nothing
// covered: the argv actually handed to `ps`, the column-to-field mapping for each dialect,
// the BSD duration grammar, and runTool's behaviour on a non-zero exit. All of them are
// hermetic — the process-table test re-execs THIS test binary rather than shelling out to
// a real `ps`, so the suite proves the same thing on a host that has no `ps` at all.

import (
	"os"
	"runtime"
	"strings"
	"testing"
	"time"
)

// psHelperEnv selects the behaviour of a re-exec'd copy of this test binary. Guarded on an
// env var so TestProcguardHelperProcess is an inert pass in an ordinary run.
const psHelperEnv = "PROCGUARD_RUNTOOL_HELPER"

// psHelperStderr is the sentence BSD `ps` prints (to stderr) before exiting 1 on a keyword
// it does not know — the sentence the pre-#5385 runTool discarded along with the table.
const psHelperStderr = "ps: nlwp: keyword not found"

// TestProcguardHelperProcess is the subprocess body runTool executes. It is not a real
// test: it dispatches on psHelperEnv and exits, so it stays an inert pass when the env is
// unset (the normal suite run).
func TestProcguardHelperProcess(t *testing.T) {
	switch os.Getenv(psHelperEnv) {
	case "":
		return // ordinary `go test` — inert.
	case "table-then-fail":
		// The exact shape #5385 describes: a COMPLETE, valid table on stdout, a complaint
		// on stderr, and a non-zero exit. Nothing about this output is unusable.
		os.Stdout.WriteString(bsdCensusTable)
		os.Stderr.WriteString(psHelperStderr + "\n")
		os.Exit(1)
	case "silent-fail":
		// The other half of the rule: a tool that produced nothing and failed. The error
		// must survive, because zero rows and no error would read as a quiet host.
		os.Stderr.WriteString("ps: command not found\n")
		os.Exit(127)
	}
}

// bsdCensusTable is a macOS `ps -eo pid=,rss=,time=,comm=` table: rss in KB, a FORMATTED
// cumulative CPU time, and comm as an absolute path (BSD prints the whole executable path
// where procps prints the short accounting name).
const bsdCensusTable = "    1  12800   0:04.21 /sbin/launchd\n" +
	"  431   4096  12:31.09 /usr/sbin/cfprefsd\n" +
	" 9182 2097152 1-02:03:04 /usr/local/bin/llama-cli\n"

// gnuCensusTable is the Linux `ps -eo pid=,nlwp=,rss=,cputimes=,comm=` table the shipped
// collector reads today: a thread count BSD cannot supply, and CPU time as whole seconds.
const gnuCensusTable = "    1   1  12800      4 systemd\n" +
	"  431   9   4096    751 dbus-daemon\n" +
	" 9182 129427 2097152  93784 llama-cli\n"

func TestRunToolKeepsStdoutOnNonZeroExit(t *testing.T) {
	t.Setenv(psHelperEnv, "table-then-fail")
	out, errText := runTool(60*time.Second, os.Args[0], "-test.run=^TestProcguardHelperProcess$")

	if !strings.Contains(out, "llama-cli") {
		t.Fatalf("stdout was discarded on a non-zero exit — the #5385 defect. out=%q err=%q", out, errText)
	}
	if errText == "" {
		t.Fatalf("a non-zero exit must still be REPORTED (the caller decides what it means); got out=%q", out)
	}
	if !strings.Contains(errText, psHelperStderr) {
		t.Fatalf("the tool's own complaint must ride the error text — %q alone names nothing; got %q",
			"exit status 1", errText)
	}
	// The salvaged bytes must be usable, not merely present.
	procs, secs := parsePSCensus(out, psCensusSpec("darwin"))
	if len(procs) != 3 {
		t.Fatalf("salvaged table must parse to 3 rows, got %d: %+v", len(procs), procs)
	}
	if secs[9182] == 0 {
		t.Fatalf("salvaged table must carry cpu seconds for pid 9182: %+v", secs)
	}
}

func TestRunToolReportsAFailureThatProducedNothing(t *testing.T) {
	t.Setenv(psHelperEnv, "silent-fail")
	out, errText := runTool(60*time.Second, os.Args[0], "-test.run=^TestProcguardHelperProcess$")
	if strings.TrimSpace(out) != "" {
		t.Fatalf("helper produced no stdout, got %q", out)
	}
	if errText == "" {
		t.Fatalf("a failure with no output must not read as success")
	}
}

// TestCensusErrorKeepsOnlyRowlessFailures pins the invariant every consumer depends on:
// a non-empty error means there is NO census. cmd/fak/resume_stopped_liveness.go reads
// `no error AND >= 1 row` as "readable"; if this rule ever let rows and an error out
// together, that consumer would silently stop witnessing driver liveness on hosts whose
// census had just been fixed.
func TestCensusErrorKeepsOnlyRowlessFailures(t *testing.T) {
	if got := censusError(0, "exit status 1"); got != "exit status 1" {
		t.Fatalf("a failure that produced no rows must be reported, got %q", got)
	}
	if got := censusError(880, "exit status 1"); got != "" {
		t.Fatalf("a complete table plus a non-zero exit is a census, not a failure, got %q", got)
	}
	if got := censusError(0, ""); got != "" {
		t.Fatalf("a clean run that listed nothing is not an error, got %q", got)
	}
}

// TestPSArgvPinnedPerGOOS is the regression fence in BOTH directions: darwin must never
// again be handed a procps-only keyword, and linux must keep the exact invocation that
// works there today (a "portable" column set that dropped nlwp would blind the thread
// dimension this whole package exists for).
func TestPSArgvPinnedPerGOOS(t *testing.T) {
	cases := []struct {
		name string
		got  []string
		want []string
	}{
		{"linux census", psCensusSpec("linux").args, []string{"-eo", "pid=,nlwp=,rss=,cputimes=,comm="}},
		{"linux relations", psRelationSpec("linux").args, []string{"-eo", "pid=,ppid=,etimes=,comm=,args="}},
		{"darwin census", psCensusSpec("darwin").args, []string{"-eo", "pid=,rss=,time=,comm="}},
		{"darwin relations", psRelationSpec("darwin").args, []string{"-eo", "pid=,ppid=,etime=,comm=,args="}},
	}
	for _, tc := range cases {
		if len(tc.got) != len(tc.want) {
			t.Fatalf("%s argv = %q, want %q", tc.name, tc.got, tc.want)
		}
		for i := range tc.want {
			if tc.got[i] != tc.want[i] {
				t.Fatalf("%s argv = %q, want %q", tc.name, tc.got, tc.want)
			}
		}
	}
	// The keyword-level assertion, independent of the exact strings above: BSD `ps` has
	// none of these, and asking for one costs the ENTIRE census, not just that column.
	for _, spec := range []psSpec{psCensusSpec("darwin"), psRelationSpec("darwin")} {
		joined := strings.Join(spec.args, " ")
		for _, procpsOnly := range []string{"nlwp", "cputimes", "etimes"} {
			if strings.Contains(joined, procpsOnly) {
				t.Fatalf("darwin argv %q asks for the procps-only keyword %q — BSD ps rejects it and exits 1 (#5385)",
					joined, procpsOnly)
			}
		}
	}
	// An un-witnessed GOOS keeps the procps invocation deliberately (see psBSD).
	if psCensusSpec("freebsd").args[1] != psCensusSpec("linux").args[1] {
		t.Fatalf("only darwin is a witnessed BSD dialect; other GOOS must keep the procps set")
	}
}

func TestParsePSDuration(t *testing.T) {
	cases := []struct {
		in   string
		want float64
		ok   bool
	}{
		{"0", 0, true},                // procps cputimes/etimes: bare seconds
		{"93784", 93784, true},        // procps: a long-lived process
		{"0:04", 4, true},             // BSD etime: mm:ss
		{"12:31", 751, true},          // BSD etime: mm:ss past ten minutes
		{"0:04.21", 4.21, true},       // BSD time: mm:ss.ff (fractional CPU seconds)
		{"01:02:03", 3723, true},      // BSD: hh:mm:ss
		{"1-02:03:04", 93784, true},   // BSD: dd-hh:mm:ss
		{"10-00:00:00", 864000, true}, // BSD: whole days
		{" 3:20 ", 200, true},         // surrounding whitespace is ps padding
		{"", 0, false},                // absent column
		{"-", 0, false},               // ps placeholder
		{"?", 0, false},               // garbage
		{"ELAPSED", 0, false},         // a header line that leaked through
		{"1:2:3:4", 0, false},         // deeper than any dialect emits
		{"NaN", 0, false},             // ParseFloat would accept this; a census never says NaN
		{"Inf", 0, false},             // likewise
		{"1e3", 0, false},             // likewise
		{"-5", 0, false},              // a negative duration is not a measurement
		{"12:", 0, false},             // truncated
		{"/usr/bin/thing", 0, false},  // a column read out of position
	}
	for _, tc := range cases {
		got, ok := parsePSDuration(tc.in)
		if ok != tc.ok {
			t.Fatalf("parsePSDuration(%q) ok = %v, want %v (got %v)", tc.in, ok, tc.ok, got)
		}
		if ok && got != tc.want {
			t.Fatalf("parsePSDuration(%q) = %v, want %v", tc.in, got, tc.want)
		}
		if !ok && got != 0 {
			t.Fatalf("parsePSDuration(%q) rejected but returned %v — an unread column must not become a number", tc.in, got)
		}
	}
}

// TestParsePSCensusDarwin is the row-level half of the darwin fix: the BSD table maps to
// the right fields, the formatted CPU column becomes seconds, and the field BSD genuinely
// cannot supply stays NIL rather than becoming a zero (a 0 thread count is a measurement
// and would be a lie; nil is skipped by Classify).
func TestParsePSCensusDarwin(t *testing.T) {
	procs, secs := parsePSCensus(bsdCensusTable, psCensusSpec("darwin"))
	if len(procs) != 3 {
		t.Fatalf("want 3 rows, got %d: %+v", len(procs), procs)
	}
	if procs[0].PID != 1 || procs[0].Name != "launchd" {
		t.Fatalf("pid/name mapped wrong: %+v", procs[0])
	}
	if procs[2].Name != "llama-cli" {
		t.Fatalf("BSD comm is an absolute path and must be reduced to a name: %+v", procs[2])
	}
	if procs[2].WSMB == nil || *procs[2].WSMB != 2048 {
		t.Fatalf("rss KB must fold to MB: %+v", procs[2].WSMB)
	}
	for _, p := range procs {
		if p.Threads != nil {
			t.Fatalf("BSD ps has no thread-count keyword — pid %d must report nil, not %d", p.PID, *p.Threads)
		}
	}
	// mm:ss.ff carries a fraction, so compare within a tick rather than bit-for-bit.
	near := func(got, want float64) bool { return got-want < 0.001 && want-got < 0.001 }
	if !near(secs[1], 4.21) || !near(secs[431], 751.09) || !near(secs[9182], 93784) {
		t.Fatalf("formatted BSD cpu times must fold to seconds: %+v", secs)
	}
}

// TestParsePSCensusLinuxUnchanged is the no-regression half: the Linux dialect keeps
// emitting every field it emitted before the darwin branch existed.
func TestParsePSCensusLinuxUnchanged(t *testing.T) {
	procs, secs := parsePSCensus(gnuCensusTable, psCensusSpec("linux"))
	if len(procs) != 3 {
		t.Fatalf("want 3 rows, got %d: %+v", len(procs), procs)
	}
	last := procs[2]
	if last.Name != "llama-cli" || last.Threads == nil || *last.Threads != 129427 {
		t.Fatalf("linux must still carry the thread count: %+v", last)
	}
	if last.WSMB == nil || *last.WSMB != 2048 {
		t.Fatalf("linux must still carry the working set: %+v", last.WSMB)
	}
	if secs[9182] != 93784 || secs[1] != 4 {
		t.Fatalf("linux cputimes seconds must survive: %+v", secs)
	}
	// The one-field-short tolerance the pre-#5385 parser had: the cpu column is absent and
	// the name shifts left into its place. The row still counts; only its CPU time is unknown.
	short, shortSecs := parsePSCensus("  77   3   512 kworker\n", psCensusSpec("linux"))
	if len(short) != 1 || short[0].Name != "kworker" || short[0].Threads == nil || *short[0].Threads != 3 {
		t.Fatalf("a one-short line must still yield its row: %+v", short)
	}
	if _, ok := shortSecs[77]; ok {
		t.Fatalf("a row with no cpu column must have no cpu seconds, not 0")
	}
}

func TestParsePSRelationsBothDialects(t *testing.T) {
	// darwin: etime is formatted, and args carries spaces the split must not eat.
	darwin := " 9182     1    1-02:03:04 /usr/local/bin/claude claude --resume abc-123 --dangerously-skip\n"
	got := parsePSRelations(darwin, psRelationSpec("darwin"))
	if len(got) != 1 {
		t.Fatalf("want 1 row, got %d: %+v", len(got), got)
	}
	if got[0].PID != 9182 || got[0].PPID == nil || *got[0].PPID != 1 {
		t.Fatalf("pid/ppid mapped wrong: %+v", got[0])
	}
	if got[0].AgeSec == nil || *got[0].AgeSec != 93784 {
		t.Fatalf("BSD etime must fold to seconds — a nil or 0 age here is what made the "+
			"stale-child reap and the resume ceiling inert: %+v", got[0].AgeSec)
	}
	if !strings.Contains(got[0].Cmdline, "--resume abc-123") {
		t.Fatalf("the whole command line must survive the split: %q", got[0].Cmdline)
	}
	if got[0].Name != "claude" {
		t.Fatalf("name must be the basename of comm: %q", got[0].Name)
	}

	// linux: etimes is already seconds, unchanged from what shipped.
	linux := " 9182     1  93784 claude claude --resume abc-123\n"
	gotLinux := parsePSRelations(linux, psRelationSpec("linux"))
	if len(gotLinux) != 1 || gotLinux[0].AgeSec == nil || *gotLinux[0].AgeSec != 93784 {
		t.Fatalf("linux etimes seconds must survive: %+v", gotLinux)
	}

	// A row with no argv at all (a zombie keeps its accounting name and nothing else):
	// Cmdline falls back to the command name, as it did before.
	zombie := " 4242  1  17 sleep\n"
	gotZombie := parsePSRelations(zombie, psRelationSpec("linux"))
	if len(gotZombie) != 1 || gotZombie[0].Cmdline != "sleep" {
		t.Fatalf("an argv-less row must still list, naming what it can: %+v", gotZombie)
	}
}

// TestCollectPOSIXCensusIsNonEmpty is the only test here that needs a real host: it runs
// the shipped collector against the machine's own `ps`. It is the check that would have
// caught #5385 on a Mac, and it is honestly skipped where it cannot mean anything — the
// pure tests above are what carry the darwin claim from a Windows or Linux runner.
func TestCollectPOSIXCensusIsNonEmpty(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX collector: this host has no ps (the Windows collectors are exercised by the CIM/Get-Process path)")
	}
	procs, errStr := collectPOSIX()
	if errStr != "" {
		t.Fatalf("resource census failed on a live host: %s", errStr)
	}
	if len(procs) == 0 {
		t.Fatalf("a live host always runs at least this test binary — an empty census is the #5385 defect")
	}
	rel, relErr := collectPOSIXRelations()
	if relErr != "" {
		t.Fatalf("relations census failed on a live host: %s", relErr)
	}
	self := os.Getpid()
	for _, p := range rel {
		if p.PID == self {
			return // the census can see its own reader: it is complete enough to trust
		}
	}
	t.Fatalf("relations census of %d rows did not contain its own reader (pid %d)", len(rel), self)
}
