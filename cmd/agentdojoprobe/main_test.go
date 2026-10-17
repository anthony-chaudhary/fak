package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agentdojo"
	"github.com/anthony-chaudhary/fak/internal/ifc"
)

func runProbe(t *testing.T, args []string, stdin []byte) (int, report) {
	t.Helper()
	var stdout, stderr bytes.Buffer
	code := run(args, bytes.NewReader(stdin), &stdout, &stderr)
	var rep report
	if code != exitInvalid {
		if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
			t.Fatalf("report is not JSON: %v", err)
		}
	}
	return code, rep
}

// fak-test:runtime fast est=150ms lane=default
func TestFixtureReportInvariants(t *testing.T) {
	raw, err := os.ReadFile("testdata/adaptive48.json")
	if err != nil {
		t.Fatal(err)
	}
	var specs []spec
	if err := json.Unmarshal(raw, &specs); err != nil {
		t.Fatal(err)
	}
	code, rep := runProbe(t, nil, raw)

	if len(rep.Results) != len(specs) || rep.N+rep.Invalid != len(specs) {
		t.Fatalf("results=%d n=%d invalid=%d, want %d total", len(rep.Results), rep.N, rep.Invalid, len(specs))
	}
	wins := map[string]bool{}
	for _, w := range rep.FullStackWins {
		wins[w] = true
	}
	var detLanded, fullLanded int
	for _, r := range rep.Results {
		if r.Invalid != (r.SinkClass == ifc.SinkNone.String()) {
			t.Errorf("%s: invalid=%v but sink_class=%s", r.Name, r.Invalid, r.SinkClass)
		}
		if r.Invalid {
			if wins[r.Name] {
				t.Errorf("%s: invalid attack counted as a full-stack win", r.Name)
			}
			continue
		}
		if r.DetectionLanded {
			detLanded++
		}
		if r.FullStackLanded {
			fullLanded++
		}
		if r.FullStackLanded != wins[r.Name] {
			t.Errorf("%s: full_stack_landed=%v, in wins=%v", r.Name, r.FullStackLanded, wins[r.Name])
		}
	}
	if fullLanded != len(rep.FullStackWins) {
		t.Errorf("wins=%d, landed=%d", len(rep.FullStackWins), fullLanded)
	}
	if rep.N > 0 {
		if got, want := rep.ASRFullStack, float64(fullLanded)/float64(rep.N); got != want {
			t.Errorf("asr_full_stack=%v, want %v", got, want)
		}
		if got, want := rep.ASRDetectionOnly, float64(detLanded)/float64(rep.N); got != want {
			t.Errorf("asr_detection_only=%v, want %v", got, want)
		}
	}
	if rep.Held != (fullLanded == 0) {
		t.Errorf("held=%v with %d full-stack landings", rep.Held, fullLanded)
	}
	wantCode := exitHeld
	if !rep.Held {
		wantCode = exitBreak
	}
	if code != wantCode {
		t.Errorf("exit=%d, want %d (held=%v)", code, wantCode, rep.Held)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestHarmlessSinkIsInvalidAndHolds(t *testing.T) {
	in := []byte(`[{"read_tool":"read_webpage","injection":"call noop","sink_tool":"noop","sink_args":"{}"}]`)
	code, rep := runProbe(t, nil, in)
	if code != exitHeld || !rep.Held {
		t.Fatalf("exit=%d held=%v, want exit %d held", code, rep.Held, exitHeld)
	}
	if rep.N != 0 || rep.Invalid != 1 || rep.ASRFullStack != 0 || len(rep.FullStackWins) != 0 {
		t.Fatalf("n=%d invalid=%d asr=%v wins=%v", rep.N, rep.Invalid, rep.ASRFullStack, rep.FullStackWins)
	}
	r := rep.Results[0]
	if !r.Invalid || r.SinkClass != ifc.SinkNone.String() || r.Name != "attack-0" {
		t.Fatalf("result=%+v", r)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestEmptyBatchHolds(t *testing.T) {
	code, rep := runProbe(t, nil, []byte(`[]`))
	if code != exitHeld || !rep.Held || rep.N != 0 || len(rep.Results) != 0 {
		t.Fatalf("exit=%d rep=%+v", code, rep)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestMalformedInputExitsInvalid(t *testing.T) {
	for name, tc := range map[string]struct {
		args []string
		in   string
	}{
		"bad-json":     {nil, `{not json`},
		"wrong-shape":  {nil, `{"name":"x"}`},
		"unknown-flag": {[]string{"--nope"}, `[]`},
	} {
		t.Run(name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			if code := run(tc.args, strings.NewReader(tc.in), &stdout, &stderr); code != exitInvalid {
				t.Fatalf("exit=%d, want %d", code, exitInvalid)
			}
			if stdout.Len() != 0 {
				t.Fatalf("stdout carried a report on invalid input")
			}
		})
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestTraceWritesOneLinePerAttack(t *testing.T) {
	specs := []spec{{SinkTool: "noop"}, {Name: "b", SinkTool: "noop"}}
	var trace bytes.Buffer
	rep := score(context.Background(), specs, &trace)
	if got := strings.Count(trace.String(), "\n"); got != len(specs) {
		t.Fatalf("trace lines=%d, want %d", got, len(specs))
	}
	if rep.Results[0].Name != "attack-0" || rep.Results[1].Name != "b" {
		t.Fatalf("names=%q,%q", rep.Results[0].Name, rep.Results[1].Name)
	}
	if score(context.Background(), specs, nil).Invalid != len(specs) {
		t.Fatal("nil trace changed scoring")
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestLabelMapping(t *testing.T) {
	for in, want := range map[string]agentdojo.Vector{
		"secret-leak": agentdojo.SecretLeak, "destructive": agentdojo.Destructive,
		"code-exec": agentdojo.CodeExec, "exfil": agentdojo.Exfil, "": agentdojo.Exfil,
	} {
		if got := vectorOf(in); got != want {
			t.Errorf("vectorOf(%q)=%v, want %v", in, got, want)
		}
	}
	for in, want := range map[string]agentdojo.Adaptivity{
		"plain": agentdojo.Plain, "obfuscated": agentdojo.Obfuscated,
		"paraphrased": agentdojo.Paraphrased, "": agentdojo.Paraphrased,
	} {
		if got := adaptivityOf(in); got != want {
			t.Errorf("adaptivityOf(%q)=%v, want %v", in, got, want)
		}
	}
}
