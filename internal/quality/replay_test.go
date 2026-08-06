package quality

import (
	"encoding/json"
	"strings"
	"testing"
)

// plantedBundle runs the demo case against an injected defect and returns the
// scrubbed failure bundle the spine emitted, marshalled to JSON exactly as
// `fak quality run --json` would store it. It is the "planted representative
// defect" half of this issue's witness: everything downstream of here sees only
// the serialized artifact.
func plantedBundle(t *testing.T, defect string) (FailureBundle, []byte) {
	t.Helper()
	c := DemoCase()
	res, err := RunCase(c, ReferenceRunner{}, DemoEngine(defect), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase(%s): %v", defect, err)
	}
	if res.Pass || res.FailureBundle == nil {
		t.Fatalf("injected %q defect must fail and emit a bundle; got %s", defect, Explain(res))
	}
	blob, err := json.Marshal(res.FailureBundle)
	if err != nil {
		t.Fatalf("marshal bundle: %v", err)
	}
	return *res.FailureBundle, blob
}

// TestReplayReproducesPlantedDefectFromBundleAlone is the #4515 round trip: for
// every injected defect the spine can plant, the emitted bundle — reloaded from
// its own JSON, with no case file, environment, or live engine in play —
// reproduces the SAME failing oracle at the SAME first divergence.
func TestReplayReproducesPlantedDefectFromBundleAlone(t *testing.T) {
	for _, defect := range []string{"decode", "stop", "report"} {
		t.Run(defect, func(t *testing.T) {
			original, blob := plantedBundle(t, defect)

			loaded, err := LoadBundle(blob)
			if err != nil {
				t.Fatalf("LoadBundle: %v", err)
			}
			v := Replay(loaded)
			if !v.Reproduced {
				t.Fatalf("bundle must replay to its recorded failure; got %s", ExplainReplay(v))
			}
			if v.Inconclusive {
				t.Errorf("a reproduced replay is not inconclusive: %s", v.Reason)
			}
			if v.Schema != ReplaySchema {
				t.Errorf("verdict schema = %q, want %q", v.Schema, ReplaySchema)
			}
			if v.Observed == nil {
				t.Fatal("a reproduced replay must record what it observed")
			}
			if got, want := v.Observed.FailingOracle, original.FailingOracle; got != want {
				t.Errorf("replayed failing oracle = %q, want %q", got, want)
			}
			if got, want := v.Observed.FailingKind, original.FailingKind; got != want {
				t.Errorf("replayed failing kind = %q, want %q", got, want)
			}
			switch {
			case original.FirstDivergence == nil && v.Observed.FirstDivergence != nil:
				t.Errorf("replay invented a divergence the bundle never recorded: %+v", *v.Observed.FirstDivergence)
			case original.FirstDivergence != nil && v.Observed.FirstDivergence == nil:
				t.Error("replay lost the bundle's recorded first divergence")
			case original.FirstDivergence != nil:
				if *v.Observed.FirstDivergence != *original.FirstDivergence {
					t.Errorf("replayed divergence = %+v, want %+v", *v.Observed.FirstDivergence, *original.FirstDivergence)
				}
			}
			// The replay is itself an auditable artifact, not a bare boolean.
			if v.Result == nil {
				t.Fatal("a replay that ran must carry its replayed result")
			}
			if v.Result.Pass {
				t.Error("replaying a failure bundle must not produce a passing result")
			}
			if v.Result.Manifest.CaseID != original.CaseID {
				t.Errorf("replayed manifest case = %q, want %q", v.Result.Manifest.CaseID, original.CaseID)
			}
		})
	}
}

// TestReplayAcceptsTheResultEnvelope proves the verb consumes the artifact CI
// actually stores: the whole `run --json` Result, not a hand-unwrapped bundle.
func TestReplayAcceptsTheResultEnvelope(t *testing.T) {
	c := DemoCase()
	res, err := RunCase(c, ReferenceRunner{}, DemoEngine("decode"), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	blob, err := json.Marshal(res)
	if err != nil {
		t.Fatalf("marshal result: %v", err)
	}
	b, err := LoadBundle(blob)
	if err != nil {
		t.Fatalf("LoadBundle(result envelope): %v", err)
	}
	if v := Replay(b); !v.Reproduced {
		t.Fatalf("result-envelope bundle must replay; got %s", ExplainReplay(v))
	}
}

// TestLoadBundleRefusesAPassingResult keeps "there is nothing to replay" out of
// the verdict vocabulary: a green result is an input error, not an inconclusive
// replay, so the caller is told it handed over the wrong artifact.
func TestLoadBundleRefusesAPassingResult(t *testing.T) {
	c := DemoCase()
	res, err := RunCase(c, ReferenceRunner{}, DemoEngine(""), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	blob, err := json.Marshal(res)
	if err != nil {
		t.Fatalf("marshal result: %v", err)
	}
	if _, err := LoadBundle(blob); err == nil {
		t.Fatal("a passing result carries no bundle; LoadBundle must refuse it")
	}
}

// TestLoadBundleRefusesUnreadableArtifacts holds the reader to the package's
// pinned-schema posture: unknown fields and trailing documents are refused
// rather than silently half-decoded into a replayable-looking bundle.
func TestLoadBundleRefusesUnreadableArtifacts(t *testing.T) {
	_, blob := plantedBundle(t, "decode")
	for name, mutate := range map[string]func([]byte) []byte{
		"unknown field":    func(b []byte) []byte { return append([]byte(`{"not_a_bundle_field":1,`), b[1:]...) },
		"trailing doc":     func(b []byte) []byte { return append(append([]byte{}, b...), []byte("{}")...) },
		"not json":         func([]byte) []byte { return []byte("this is not json") },
		"truncated object": func(b []byte) []byte { return b[:len(b)/2] },
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := LoadBundle(mutate(blob)); err == nil {
				t.Fatalf("LoadBundle must refuse a %s artifact", name)
			}
		})
	}
}

// TestReplayReportsIncompleteBundlesAsInconclusive is the "missing or
// inconclusive evidence is never pass" half of the contract: every way a bundle
// can be short of replay-critical evidence is reported, with a reason, and none
// of them reproduce.
func TestReplayReportsIncompleteBundlesAsInconclusive(t *testing.T) {
	cases := map[string]struct {
		mutate func(*FailureBundle)
		want   string
	}{
		"no engine trace": {
			mutate: func(b *FailureBundle) { b.Engine = Trace{Runner: b.Engine.Runner} },
			want:   "no engine trace",
		},
		"no reference trace": {
			mutate: func(b *FailureBundle) { b.Reference = Trace{Runner: b.Reference.Runner} },
			want:   "no reference trace",
		},
		"unscrubbed": {
			mutate: func(b *FailureBundle) { b.Scrubbed = false },
			want:   "not marked scrubbed",
		},
		"no failing oracle": {
			mutate: func(b *FailureBundle) { b.FailingOracle = "" },
			want:   "no failing oracle",
		},
		"case stripped of oracles": {
			mutate: func(b *FailureBundle) { b.Case.Oracles = nil },
			want:   "not runnable",
		},
		"case id mismatch": {
			mutate: func(b *FailureBundle) { b.Case.ID = "some-other-case" },
			want:   "does not match",
		},
		"unregistered oracle": {
			mutate: func(b *FailureBundle) { b.Case.Oracles = []string{"no-such-oracle"} },
			want:   "unrunnable oracle",
		},
		"blames an oracle the case never runs": {
			mutate: func(b *FailureBundle) { b.FailingOracle = "grounding-rubric"; b.Case.Oracles = []string{"greedy-token-diff"} },
			want:   "which the embedded case does not run",
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			b, _ := plantedBundle(t, "decode")
			tc.mutate(&b)
			v := Replay(b)
			if v.Reproduced {
				t.Fatalf("an incomplete bundle must never reproduce: %s", ExplainReplay(v))
			}
			if !v.Inconclusive {
				t.Fatalf("an incomplete bundle must be reported inconclusive; got %s", ExplainReplay(v))
			}
			if !strings.Contains(v.Reason, tc.want) {
				t.Errorf("reason %q does not name the missing evidence (want substring %q)", v.Reason, tc.want)
			}
			if !strings.Contains(ExplainReplay(v), "INCONCLUSIVE") {
				t.Errorf("rendered replay must state INCONCLUSIVE:\n%s", ExplainReplay(v))
			}
		})
	}
}

// TestReplayRefusesToLaunderADriftedBundle: a bundle whose evidence no longer
// produces the failure it claims has NOT reproduced. Reproduction is judged on
// the signature, so "it failed again, somewhere else" is not a green replay —
// and neither is "it passed".
func TestReplayRefusesToLaunderADriftedBundle(t *testing.T) {
	t.Run("replays to a different divergence", func(t *testing.T) {
		b, _ := plantedBundle(t, "decode")
		// Move the planted flip to a later token: the bundle still fails, but at
		// an index its own record does not name.
		b.Engine.Tokens = append([]string(nil), b.Reference.Tokens...)
		b.Engine.Tokens[4] = "month"
		v := Replay(b)
		if v.Reproduced {
			t.Fatalf("a drifted bundle must not report reproduced: %s", ExplainReplay(v))
		}
		if v.Inconclusive {
			t.Errorf("a bundle that replayed cleanly to a DIFFERENT failure is not inconclusive: %s", v.Reason)
		}
		if !strings.Contains(v.Reason, "different failure") {
			t.Errorf("reason %q does not name the drift", v.Reason)
		}
	})
	t.Run("replays to a pass", func(t *testing.T) {
		b, _ := plantedBundle(t, "decode")
		// Evidence that no longer contains the defect: the traces agree.
		b.Engine.Tokens = append([]string(nil), b.Reference.Tokens...)
		b.Engine.Text = b.Reference.Text
		v := Replay(b)
		if v.Reproduced {
			t.Fatalf("a bundle whose evidence passes has not reproduced anything: %s", ExplainReplay(v))
		}
		if !strings.Contains(v.Reason, "did not reproduce") {
			t.Errorf("reason %q does not say the recorded failure failed to reproduce", v.Reason)
		}
	})
}

// TestReplayUsesTheBundleAndNotTheAmbientCase pins the property the whole
// artifact rests on: replay judges the bundle's OWN captured traces. Rewriting
// the embedded case's reference — while leaving the captured reference trace
// intact — must not move the verdict, because the case's copy is not what ran.
func TestReplayUsesTheBundleAndNotTheAmbientCase(t *testing.T) {
	b, _ := plantedBundle(t, "decode")
	b.Case.Reference = Trace{Tokens: []string{"totally", "different"}, Text: "totally different"}
	v := Replay(b)
	if !v.Reproduced {
		t.Fatalf("replay must judge the captured traces, not the case's reference copy; got %s", ExplainReplay(v))
	}
}
