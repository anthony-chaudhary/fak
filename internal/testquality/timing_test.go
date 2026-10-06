package testquality

import "testing"

func codesIn(t *testing.T, src string) map[string]int {
	t.Helper()
	got, err := Analyze("p/x_test.go", []byte(src))
	if err != nil {
		t.Fatal(err)
	}
	m := map[string]int{}
	for _, f := range got {
		m[f.Code]++
	}
	return m
}

func TestSleepSyncOnePerFunctionIncludingClosures(t *testing.T) {
	m := codesIn(t, `package p
import ("testing"; "time")
func TestA(t *testing.T) {
	time.Sleep(time.Millisecond)
	go func() { time.Sleep(time.Second) }()
	t.Run("x", func(t *testing.T) { time.Sleep(1) })
	if false { t.Fatal("x") }
}
func TestB(t *testing.T) { if false { t.Fatal("x") } }
`)
	if m[CodeSleepSync] != 1 {
		t.Fatalf("sleep findings=%d want 1 (one per test func): %v", m[CodeSleepSync], m)
	}
}

func TestSleepSyncIgnoresPollLoops(t *testing.T) {
	m := codesIn(t, `package p
import ("testing"; "time")
func TestPoll(t *testing.T) {
	deadline := time.Now().Add(time.Second)
	for !ready() {
		if time.Now().After(deadline) { t.Fatal("never ready") }
		time.Sleep(time.Millisecond)
	}
}
func ready() bool { return true }
`)
	if m[CodeSleepSync] != 0 {
		t.Fatalf("a deadline poll loop is the recommended fix, not the defect: %v", m)
	}
}

func TestSleepSyncFollowsImportAliasAndIgnoresShadowName(t *testing.T) {
	aliased := codesIn(t, `package p
import ("testing"; clk "time")
func TestA(t *testing.T) { clk.Sleep(1); if false { t.Fatal("x") } }
`)
	if aliased[CodeSleepSync] != 1 {
		t.Fatalf("aliased import not followed: %v", aliased)
	}
	noImport := codesIn(t, `package p
import "testing"
type clock struct{}
func (clock) Sleep(int) {}
func TestA(t *testing.T) { var time clock; time.Sleep(1); if false { t.Fatal("x") } }
`)
	if noImport[CodeSleepSync] != 0 {
		t.Fatalf("a local named time without the import must not be reported: %v", noImport)
	}
}

func TestWallclockAssertInlineAndViaVariable(t *testing.T) {
	m := codesIn(t, `package p
import ("testing"; "time")
func TestInline(t *testing.T) {
	start := time.Now()
	if time.Since(start) > time.Second { t.Fatalf("slow") }
}
func TestVar(t *testing.T) {
	start := time.Now()
	elapsed := time.Since(start)
	if elapsed > 50*time.Millisecond { t.Errorf("slow: %v", elapsed) }
}
func TestLogOnly(t *testing.T) {
	start := time.Now()
	if d := time.Since(start); d > time.Second { t.Logf("slow %v", d) }
	if false { t.Fatal("x") }
}
`)
	if m[CodeWallclockAssert] != 2 {
		t.Fatalf("wallclock findings=%d want 2 (log-only branch is not an assertion): %v", m[CodeWallclockAssert], m)
	}
}

func TestUnconditionalSkipOnlyTopLevel(t *testing.T) {
	m := codesIn(t, `package p
import "testing"
func TestDead(t *testing.T) { t.Skip("broken, fix later"); if false { t.Fatal("x") } }
func TestGuarded(t *testing.T) { if testing.Short() { t.Skip("short") }; if false { t.Fatal("x") } }
func TestSubSkip(t *testing.T) { t.Run("a", func(t *testing.T) { t.Skip("x") }) }
func TestHelperChild(t *testing.T) { for _, a := range []string{} { if a == "--" { return } }; t.Skip("helper only") }
`)
	if m[CodeUnconditionalSkip] != 1 {
		t.Fatalf("unconditional skip findings=%d want 1: %v", m[CodeUnconditionalSkip], m)
	}
}

func TestNewCodesAreKnownToBaselineParser(t *testing.T) {
	for _, c := range []string{CodeSleepSync, CodeWallclockAssert, CodeUnconditionalSkip} {
		if _, err := ParseBaseline([]byte(c + "\tp/x_test.go\tTestX\t1\n")); err != nil {
			t.Fatalf("%s rejected by baseline parser: %v", c, err)
		}
	}
}
