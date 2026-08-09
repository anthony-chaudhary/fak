package main

// serve_stream_progress_test.go — #5545: `fak serve --stream-progress-timeout`, the operator
// route to the streaming CONTENT-progress deadline (#5486).
//
// The deadline's config surface landed without a front door: agent.DefaultStreamProgressTimeout
// and HTTPPlanner.StreamProgressTimeout existed, and nothing in cmd/fak or internal/gateway set
// the field — a hard 300s deadline whose escape hatch (a NEGATIVE field value) could not be
// reached from a command line at all. These cases pin the front-door half of the route: the
// flag defaults to the constant rather than a copied literal, the operator's off switch is the
// house 0 spelling, and 0 is translated into the negative encoding the planner's resolver
// recognizes. The other half — that the Config value really is the window the stall reader
// arms, and that the negative encoding really disables it — is
// internal/gateway/stream_progress_timeout_test.go.

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// TestServeStreamProgressTimeoutDefaultsToTheAgentConstant pins the single-source-of-truth
// rule (the DefaultCtxViewBudget idiom): an operator who types nothing gets exactly the
// constant the planner would have used on its own, so the flag default and the resolver
// default can never drift into two different real deadlines.
func TestServeStreamProgressTimeoutDefaultsToTheAgentConstant(t *testing.T) {
	fs, sf := newServeFlagSet()
	if !parseFlags(fs, nil) {
		t.Fatal("parsing an empty argv must succeed")
	}
	if got := *sf.streamProgressTimeout; got != agent.DefaultStreamProgressTimeout {
		t.Fatalf("--stream-progress-timeout default = %s, want agent.DefaultStreamProgressTimeout (%s)", got, agent.DefaultStreamProgressTimeout)
	}
	// The value that actually reaches gateway.Config on an untouched boot must be that same
	// constant — not zero (which the planner would read as "unconfigured") and not a rounded
	// copy of it.
	if got := serveStreamProgressTimeout(*sf.streamProgressTimeout); got != agent.DefaultStreamProgressTimeout {
		t.Fatalf("default boot feeds gateway.Config %s, want %s", got, agent.DefaultStreamProgressTimeout)
	}
	f := fs.Lookup("stream-progress-timeout")
	if f == nil {
		t.Fatal("--stream-progress-timeout is not registered on the serve flag set")
	}
	if f.DefValue != agent.DefaultStreamProgressTimeout.String() {
		t.Fatalf("help shows default %q, want %q", f.DefValue, agent.DefaultStreamProgressTimeout.String())
	}
	// The help has to state the off spelling, or the escape hatch is reachable only by
	// reading the source — which is the state #5545 exists to end.
	for _, want := range []string{"0 to DISABLE", "5s, 600s"} {
		if !strings.Contains(f.Usage, want) {
			t.Errorf("--stream-progress-timeout help must mention %q; got: %s", want, f.Usage)
		}
	}
}

// TestServeStreamProgressTimeoutOffSpellingIsZero pins the operator spelling and its
// translation. Zero is the off switch because that is what every other serve knob with one
// spells it as (--ctx-view-budget 0, --compact-history-budget 0, --elide-result-bytes 0,
// --assume-session-turns 0, --metrics-snapshot 0); a negative duration is the CONFIG FIELD's
// encoding, not something an operator should have to type. Because the flag defaults to the
// 300s constant, a zero here is unambiguously typed — the "0 means unconfigured" ambiguity
// the raw field has does not exist at the front door.
func TestServeStreamProgressTimeoutOffSpellingIsZero(t *testing.T) {
	for _, c := range []struct {
		name string
		argv []string
		want time.Duration
	}{
		{"untouched", nil, agent.DefaultStreamProgressTimeout},
		{"an in-band window rides through", []string{"-stream-progress-timeout", "45s"}, 45 * time.Second},
		{"an out-of-band window is passed on for the resolver to refuse", []string{"-stream-progress-timeout", "601s"}, 601 * time.Second},
		{"0 is the off switch", []string{"-stream-progress-timeout", "0"}, streamProgressTimeoutOff},
		{"0s is the same off switch", []string{"-stream-progress-timeout", "0s"}, streamProgressTimeoutOff},
		{"a hand-typed negative is honored as off too", []string{"-stream-progress-timeout", "-30s"}, streamProgressTimeoutOff},
	} {
		t.Run(c.name, func(t *testing.T) {
			fs, sf := newServeFlagSet()
			if !parseFlags(fs, c.argv) {
				t.Fatalf("parsing %v must succeed", c.argv)
			}
			if got := serveStreamProgressTimeout(*sf.streamProgressTimeout); got != c.want {
				t.Fatalf("serve feeds gateway.Config %s, want %s", got, c.want)
			}
		})
	}
	// The off value has to be NEGATIVE, because that is the only thing
	// agent.(*HTTPPlanner).streamProgressWindow treats as "disabled" — zero there means
	// "take the 300s default", which is the opposite of what the operator asked for.
	// (internal/agent's TestStreamProgressWindowResolvesTheConfigField holds that contract;
	// internal/gateway's TestStreamProgressTimeoutFromConfigReachesTheStallReader proves the
	// negative survives the trip and really leaves the stream uncut.)
	if streamProgressTimeoutOff >= 0 {
		t.Fatalf("the off encoding is %s; a non-negative value would silently re-arm the 300s default", streamProgressTimeoutOff)
	}
}

// TestServeSourceWiresStreamProgressTimeoutIntoGatewayNew is the drift guard on the link
// itself. Every case above calls serveStreamProgressTimeout directly, so all of them would
// still pass if the gateway.New literal quietly stopped carrying it — which is exactly the
// state #5545 found the knob in (a config field with no route). `fak serve-wiring --check`
// catches a row whose field serve.go STOPS setting, but its coverage walk only visits fields
// serve.go already sets, so the assignment itself needs saying out loud here.
func TestServeSourceWiresStreamProgressTimeoutIntoGatewayNew(t *testing.T) {
	root := repoRootFromTest(t)
	body, err := os.ReadFile(filepath.Join(root, "cmd", "fak", "serve.go"))
	if err != nil {
		t.Fatalf("read serve.go: %v", err)
	}
	src := string(body)

	for _, want := range []string{
		// Defaulted from the constant, never a bare 300*time.Second literal.
		`fs.Duration("stream-progress-timeout", agent.DefaultStreamProgressTimeout,`,
		"StreamProgressTimeout: serveStreamProgressTimeout(*sf.streamProgressTimeout),",
	} {
		if !strings.Contains(src, want) {
			t.Fatalf("serve.go must contain %q — without it --stream-progress-timeout is parse-only", want)
		}
	}
	if !serveConfigAssignments(src)["StreamProgressTimeout"] {
		t.Fatal("serve.go's gateway.New(gateway.Config{...}) literal must set StreamProgressTimeout; without it newConfiguredHTTPPlanner gets a zero and the flag is dropped on the floor")
	}
}

// TestServeWiringAuditsStreamProgressTimeout pins the audit surface: a Config field serve.go
// sets with no servewiringData row reds `fak serve-wiring --check` as UNAUDITED, and the
// generated table in docs/serve-config.md is where an operator finds the knob.
func TestServeWiringAuditsStreamProgressTimeout(t *testing.T) {
	var row *wiringRow
	for i := range servewiringData {
		if servewiringData[i].Field == "StreamProgressTimeout" {
			row = &servewiringData[i]
			break
		}
	}
	if row == nil {
		t.Fatal("no servewiringData row covers gateway.Config.StreamProgressTimeout; serve-wiring --check reports it UNAUDITED")
	}
	if row.Flag != "--stream-progress-timeout" {
		t.Fatalf("row Flag = %q, want --stream-progress-timeout", row.Flag)
	}
	if row.Verdict != verdictWired {
		t.Fatalf("row Verdict = %q, want %q — the deadline is default-ON at 300s, not armed by a non-default flag value", row.Verdict, verdictWired)
	}
	// The row's note is where the off spelling is documented for an operator reading the
	// wiring table rather than the flag help.
	if !strings.Contains(row.Note, "--stream-progress-timeout 0") {
		t.Fatalf("row note must name the off switch; got: %s", row.Note)
	}

	root := repoRootFromTest(t)
	doc, err := os.ReadFile(filepath.Join(root, "docs", "serve-config.md"))
	if err != nil {
		t.Fatalf("read docs/serve-config.md: %v", err)
	}
	for _, want := range []string{"`" + row.Feature + "`", "--stream-progress-timeout"} {
		if !strings.Contains(string(doc), want) {
			t.Fatalf("docs/serve-config.md is missing %q — regenerate it with `fak serve-wiring --md`", want)
		}
	}
}
