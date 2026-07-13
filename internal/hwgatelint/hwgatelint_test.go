package hwgatelint

import "testing"

// TestFixture is the corpus witness: every labeled sample must scan to its
// declared verdict and Classes. This is what `fak hwgate-lint --self-test`
// re-derives at the CLI.
func TestFixture(t *testing.T) {
	cases, passed := RunFixture()
	if passed != len(cases) {
		for _, c := range cases {
			if !c.OK {
				t.Errorf("fixture %q: expect_clean=%v got=%s classes=%v", c.Name, c.Clean, c.Got, c.Classes)
			}
		}
	}
}

func TestScanClassifies(t *testing.T) {
	tests := []struct {
		name string
		text string
		want Class
	}{
		{"no-gpu", "There is no GPU on this host, so I can't run it.", NoLocalGPU},
		{"needs-cuda", "The test needs CUDA and none is available.", NoLocalGPU},
		{"requires-accelerator", "This requires an accelerator we don't have locally.", NoLocalGPU},
		{"no-nodejs", "Blocked: the harness needs Node.js which isn't installed here.", NoLocalRuntime},
		{"local-laptop", "Can't reproduce that on this laptop.", LocalBoundary},
		{"not-enough-ram", "Not enough memory to load the model on this box.", LocalBoundary},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rep := Scan(tc.text)
			if rep.Classes[tc.want] == 0 {
				t.Fatalf("text %q: want class %s, got %+v", tc.text, tc.want, rep.Classes)
			}
			if rep.Verdict != HardwareGated {
				t.Fatalf("text %q: want verdict %s, got %s", tc.text, HardwareGated, rep.Verdict)
			}
			if rep.Redirect != Redirect {
				t.Errorf("text %q: a gated report must carry the fixed Redirect", tc.text)
			}
		})
	}
}

// TestSanctionedRouteSuppresses: a turn that already named the lab route did the
// right thing — even though it mentions "no GPU" — and must scan clean. This is
// the no-false-positive guard that makes the soak-first rollout safe.
func TestSanctionedRouteSuppresses(t *testing.T) {
	clean := []string{
		"The device witness needs a GPU this laptop lacks, so I dispatched it to the DGX via dgxbridge.",
		"No local GPU here; produced the ready-to-run command sequence and handed it to the operator.",
		"Ran it on fak-realmodel instead — no CUDA on this box, but the L4 node has it.",
		"Can't run the CUDA suite locally, so I queued it on the nightrun pipeline.",
	}
	for _, txt := range clean {
		rep := Scan(txt)
		if rep.Verdict != Clean {
			t.Errorf("redirected turn should be clean, got %s for %q", rep.Verdict, txt)
		}
		if !rep.Suppressed {
			t.Errorf("redirected turn should be marked Suppressed: %q", txt)
		}
	}
}

// TestNoBlockerNoFinding: naming GPU/CUDA without a blocker framing is not the
// anti-pattern — a plain status line must not trip the sensor.
func TestNoBlockerNoFinding(t *testing.T) {
	ok := []string{
		"The GPU node produced the witness and I recorded it.",
		"Wired the CUDA kernel path; the accelerator loop is faster.",
	}
	for _, txt := range ok {
		if rep := Scan(txt); rep.Verdict != Clean {
			t.Errorf("non-blocker hardware mention should be clean, got %s for %q", rep.Verdict, txt)
		}
	}
}

func TestCleanOutputHasNoFindings(t *testing.T) {
	rep := Scan("Implemented the parser, committed abc123, tests pass, pushed.")
	if rep.Verdict != Clean || rep.Count != 0 {
		t.Errorf("clean output: want clean/0, got %s/%d %+v", rep.Verdict, rep.Count, rep.Findings)
	}
}
