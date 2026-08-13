package main

// serve_native_help_test.go — #6659: the `--native` help text is a CONTRACT, and it had
// gone stale.
//
// `--native` shipped (#1316) owning only the buffered turn, and its help closed with "A
// streaming request falls through to the proxy path." Native streaming then landed
// (#1837/#5148): internal/gateway/messages.go routes `s.native && req.Stream` into
// serveNativeMessagesStream, which drives agent.RunArmStream and renders the owned loop's
// deltas as Anthropic SSE. The help was never migrated, so `fak serve --help` told every
// operator evaluating the native harness that half of it was still proxied.
//
// These cases capture the rendered help and hold three things: the stale claim is gone,
// the real streaming ownership is stated, and the real fallback boundary is named. The
// last is the one that keeps the fix honest — there IS a fallback, it just is not the
// proxy: a non-flushable writer, a planner without streaming, or an armed answer stop-gate
// degrade the turn to the BUFFERED NATIVE handler. Saying "no fallback" would be a second
// wrong help text. TestNativeHelpMatchesTheGatewayRouting binds each claim back to the
// source that implements it, so the next routing change reds this test rather than
// silently re-staling the help.

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// nativeFlagUsage returns the live `--native` help string off the real serve flag set.
func nativeFlagUsage(t *testing.T) string {
	t.Helper()
	fs, _ := newServeFlagSet()
	f := fs.Lookup("native")
	if f == nil {
		t.Fatal("--native is not registered on the serve flag set")
	}
	return f.Usage
}

// TestServeNativeHelpDropsTheStaleProxyFallthroughClaim is the captured-render half: it
// asserts against the help an operator actually SEES (fs.PrintDefaults, the same wall
// `fak serve --help` prints), not just the Usage field, so a claim moved into a different
// flag's text or a custom usage function cannot smuggle it back onto the terminal.
func TestServeNativeHelpDropsTheStaleProxyFallthroughClaim(t *testing.T) {
	fs, _ := newServeFlagSet()
	var rendered bytes.Buffer
	fs.SetOutput(&rendered)
	fs.PrintDefaults()
	help := rendered.String()

	if !strings.Contains(help, "-native") {
		t.Fatal("rendered serve help does not mention -native at all")
	}
	// The exact sentence #6659 exists to kill, plus the looser shapes it could be
	// reworded into while staying just as wrong.
	for _, stale := range []string{
		"A streaming request falls through to the proxy path",
		"streaming request falls through to the proxy",
		"falls through to the proxy path",
	} {
		if strings.Contains(help, stale) {
			t.Errorf("serve help still claims %q; native streaming has shipped (gateway.messages.go routes stream+native to serveNativeMessagesStream)", stale)
		}
	}
}

// TestServeNativeHelpStatesTheStreamingContract pins what the help must now SAY. A pure
// deletion of the stale sentence would pass the test above while leaving an operator with
// no statement at all about streaming — the same evaluation gap, quieter.
func TestServeNativeHelpStatesTheStreamingContract(t *testing.T) {
	usage := nativeFlagUsage(t)

	// The ownership claim: streaming is driven by the owned loop, named by the symbol an
	// operator can grep for.
	for _, want := range []string{"RunArmStream", "SSE"} {
		if !strings.Contains(usage, want) {
			t.Errorf("--native help must name %q — the streamed turn is driven by the owned loop, not the proxy; got: %s", want, usage)
		}
	}
	// The fallback boundary, stated as what it really is. Every one of these degrades to
	// the BUFFERED NATIVE handler; none of them reaches the proxy.
	for _, want := range []string{"flush", "stop-gate", "buffered"} {
		if !strings.Contains(strings.ToLower(usage), want) {
			t.Errorf("--native help must name the real fallback boundary (%q); got: %s", want, usage)
		}
	}
	// The one thing that stayed true across #1837: arming --native leaves the proxy path
	// untouched for everyone who did not.
	if !strings.Contains(usage, "byte-for-byte unchanged") {
		t.Errorf("--native help must keep the off-by-default proxy guarantee; got: %s", usage)
	}
}

// TestNativeHelpMatchesTheGatewayRouting is the drift guard. The cases above only read a
// string, so they would all still pass if the routing regressed to proxying streams —
// leaving the help newly wrong in the opposite direction. This binds each help claim to
// the source line that makes it true.
func TestNativeHelpMatchesTheGatewayRouting(t *testing.T) {
	root := repoRootFromTest(t)
	read := func(parts ...string) string {
		body, err := os.ReadFile(filepath.Join(append([]string{root}, parts...)...))
		if err != nil {
			t.Fatalf("read %v: %v", parts, err)
		}
		return string(body)
	}

	// A streamed request under --native is handed to the native streaming handler BEFORE
	// the proxy path is reached. This is the claim the stale help contradicted.
	messages := read("internal", "gateway", "messages.go")
	if !strings.Contains(messages, "if s.native && req.Stream {") ||
		!strings.Contains(messages, "s.serveNativeMessagesStream(w, r, req, reqTrace)") {
		t.Fatal("internal/gateway/messages.go no longer routes a streamed native request to serveNativeMessagesStream — the --native help's streaming claim is now wrong")
	}

	// And the fallbacks inside that handler land on the BUFFERED NATIVE turn, which is
	// why the help names serveNativeMessages' behavior rather than the proxy's.
	nativeServe := read("internal", "gateway", "native_serve.go")
	if !strings.Contains(nativeServe, "func (s *Server) serveNativeMessagesStream(") {
		t.Fatal("serveNativeMessagesStream is gone from internal/gateway/native_serve.go")
	}
	if strings.Count(nativeServe, "s.serveNativeMessages(w, r, req, reqTrace)") < 2 {
		t.Fatal("the streaming handler's non-flushable-writer / no-streaming-planner fallbacks no longer degrade to the buffered NATIVE turn; re-check what the --native help promises")
	}
	// Pinned to the call and its receiver, not to the argument spelling: the seed the loop
	// is handed is refactored independently of who drives the stream (it went task ->
	// seed.Task with the native wire), and that is not a fact this help text asserts.
	if !strings.Contains(nativeServe, "agent.RunArmStream(ctx, s.planner,") {
		t.Fatal("runNativeArmStream no longer calls agent.RunArmStream — the help names it as the streamed driver")
	}
	if !strings.Contains(nativeServe, "if s.stopGate != nil {") {
		t.Fatal("the stop-gate buffering branch is gone from runNativeArmStream; the --native help names it as a fallback into the buffered turn")
	}
}
