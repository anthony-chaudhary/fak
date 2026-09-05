package newmodel

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modeldescriptor"
)

// TestDeterminismRefusalSafeModelIntake proves that compiling pinned manifests,
// evaluating typed refusals, generating obligation graphs, executing replay corpora,
// and producing scaffolds are bitwise and deeply deterministic across successive
// invocations and under concurrent execution (race witness).
func TestDeterminismRefusalSafeModelIntake(t *testing.T) {
	validRaw := fixture(t, "qwen38-valid.json")
	envelope := nativeLaunchEnvelopeFixture(t)

	t.Run("ValidManifestIntake", func(t *testing.T) {
		firstPacket, err1 := CompileManifest(validRaw)
		if err1 != nil {
			t.Fatalf("first CompileManifest failed: %v", err1)
		}
		secondPacket, err2 := CompileManifest(validRaw)
		if err2 != nil {
			t.Fatalf("second CompileManifest failed: %v", err2)
		}
		if !reflect.DeepEqual(firstPacket, secondPacket) {
			t.Fatalf("CompileManifest outputs are not deeply equal:\nfirst:  %+v\nsecond: %+v", firstPacket, secondPacket)
		}

		firstJSON, err1 := CompileManifestJSON(validRaw)
		if err1 != nil {
			t.Fatalf("first CompileManifestJSON failed: %v", err1)
		}
		secondJSON, err2 := CompileManifestJSON(validRaw)
		if err2 != nil {
			t.Fatalf("second CompileManifestJSON failed: %v", err2)
		}
		if !bytes.Equal(firstJSON, secondJSON) {
			t.Fatal("CompileManifestJSON outputs are not byte-identical")
		}
	})

	t.Run("RefusalSafeIntake", func(t *testing.T) {
		refusalCases := []struct {
			name       string
			manifest   func() []byte
			wantReason RefusalReason
			wantAxis   string
		}{
			{
				name: "UnknownSemanticDelta",
				manifest: func() []byte {
					return fixture(t, "qwen38-unknown-delta.json")
				},
				wantReason: RefusalUnknownSemanticDelta,
				wantAxis:   "attention",
			},
			{
				name: "ContradictorySemanticDelta",
				manifest: func() []byte {
					return fixture(t, "qwen38-contradictory-delta.json")
				},
				wantReason: RefusalContradictorySemantic,
				wantAxis:   "attention",
			},
			{
				name: "MissingRollbackAction",
				manifest: func() []byte {
					var m ReleaseManifest
					if err := json.Unmarshal(validRaw, &m); err != nil {
						t.Fatal(err)
					}
					m.Rollback = ""
					raw, _ := json.Marshal(m)
					return raw
				},
				wantReason: RefusalManifestInvalid,
				wantAxis:   "rollback",
			},
			{
				name: "InvalidSourceRevision",
				manifest: func() []byte {
					var m ReleaseManifest
					if err := json.Unmarshal(validRaw, &m); err != nil {
						t.Fatal(err)
					}
					m.Source.Revision = "not-a-hex-revision"
					raw, _ := json.Marshal(m)
					return raw
				},
				wantReason: RefusalPinInvalid,
				wantAxis:   "source.revision",
			},
			{
				name: "MissingSourceURI",
				manifest: func() []byte {
					var m ReleaseManifest
					if err := json.Unmarshal(validRaw, &m); err != nil {
						t.Fatal(err)
					}
					m.Source.URI = ""
					raw, _ := json.Marshal(m)
					return raw
				},
				wantReason: RefusalPinInvalid,
				wantAxis:   "source.uri",
			},
			{
				name: "NegativeCouplingBudget",
				manifest: func() []byte {
					var m ReleaseManifest
					if err := json.Unmarshal(validRaw, &m); err != nil {
						t.Fatal(err)
					}
					m.Coupling.Budget.CoreSwitches = -1
					raw, _ := json.Marshal(m)
					return raw
				},
				wantReason: RefusalManifestInvalid,
				wantAxis:   "coupling.budget.core_switches",
			},
			{
				name: "InvalidDescriptorStateGeometry",
				manifest: func() []byte {
					var m ReleaseManifest
					if err := json.Unmarshal(validRaw, &m); err != nil {
						t.Fatal(err)
					}
					m.Descriptor.State = []modeldescriptor.Geometry{
						{Kind: "bad-state", Shape: []int{0}, BytesPerElement: 2},
					}
					raw, _ := json.Marshal(m)
					return raw
				},
				wantReason: RefusalDescriptorInvalid,
				wantAxis:   "descriptor.state",
			},
			{
				name: "IncompleteObligations",
				manifest: func() []byte {
					var m ReleaseManifest
					if err := json.Unmarshal(validRaw, &m); err != nil {
						t.Fatal(err)
					}
					// Remove performance obligation
					filtered := m.Obligations[:0]
					for _, ob := range m.Obligations {
						if ob.Kind != "performance" {
							filtered = append(filtered, ob)
						}
					}
					m.Obligations = filtered
					raw, _ := json.Marshal(m)
					return raw
				},
				wantReason: RefusalObligationsIncomplete,
				wantAxis:   "performance",
			},
		}

		for _, tc := range refusalCases {
			t.Run(tc.name, func(t *testing.T) {
				raw := tc.manifest()

				firstPacket, err1 := CompileManifest(raw)
				secondPacket, err2 := CompileManifest(raw)

				if err1 == nil || err2 == nil {
					t.Fatalf("expected refusal error; got err1=%v, err2=%v", err1, err2)
				}
				var ref1, ref2 *Refusal
				if !errors.As(err1, &ref1) || !errors.As(err2, &ref2) {
					t.Fatalf("expected *Refusal; got err1=%T(%v), err2=%T(%v)", err1, err1, err2, err2)
				}

				if !reflect.DeepEqual(ref1, ref2) {
					t.Fatalf("refusals are not deeply equal:\nref1 = %+v\nref2 = %+v", ref1, ref2)
				}
				if ref1.Reason != tc.wantReason || ref1.Axis != tc.wantAxis {
					t.Fatalf("got refusal reason=%s axis=%s; want reason=%s axis=%s", ref1.Reason, ref1.Axis, tc.wantReason, tc.wantAxis)
				}
				if !reflect.DeepEqual(firstPacket, secondPacket) {
					t.Fatalf("packets on refusal are not deeply equal:\np1 = %+v\np2 = %+v", firstPacket, secondPacket)
				}

				// Verify JSON compile produces byte-identical error results
				jraw1, jerr1 := CompileManifestJSON(raw)
				jraw2, jerr2 := CompileManifestJSON(raw)
				if jerr1 == nil || jerr2 == nil {
					t.Fatalf("expected JSON refusal error; got jerr1=%v, jerr2=%v", jerr1, jerr2)
				}
				var jref1, jref2 *Refusal
				if !errors.As(jerr1, &jref1) || !errors.As(jerr2, &jref2) {
					t.Fatalf("expected *Refusal from JSON compile; got %T, %T", jerr1, jerr2)
				}
				if !reflect.DeepEqual(jref1, jref2) {
					t.Fatalf("JSON refusals are not deeply equal:\njref1 = %+v\njref2 = %+v", jref1, jref2)
				}
				if !bytes.Equal(jraw1, jraw2) {
					t.Fatal("CompileManifestJSON output bytes on refusal are not identical")
				}
			})
		}
	})

	t.Run("NativeObligationGraph", func(t *testing.T) {
		packet, err := CompileManifest(validRaw)
		if err != nil {
			t.Fatal(err)
		}

		firstGraph, err1 := CompileNativeObligationGraph(packet, envelope)
		if err1 != nil {
			t.Fatalf("first CompileNativeObligationGraph failed: %v", err1)
		}
		secondGraph, err2 := CompileNativeObligationGraph(packet, envelope)
		if err2 != nil {
			t.Fatalf("second CompileNativeObligationGraph failed: %v", err2)
		}
		if !reflect.DeepEqual(firstGraph, secondGraph) {
			t.Fatalf("CompileNativeObligationGraph outputs are not deeply equal:\nfirst:  %+v\nsecond: %+v", firstGraph, secondGraph)
		}

		firstJSON, err1 := MarshalNativeObligationGraph(firstGraph)
		if err1 != nil {
			t.Fatal(err1)
		}
		secondJSON, err2 := MarshalNativeObligationGraph(secondGraph)
		if err2 != nil {
			t.Fatal(err2)
		}
		if !bytes.Equal(firstJSON, secondJSON) {
			t.Fatal("MarshalNativeObligationGraph bytes are not identical")
		}

		// Obligation graph refusal determinism
		refusalEnvelope := envelope
		refusalEnvelope.Backend = "unsupported-backend"
		rg1, rerr1 := CompileNativeObligationGraph(packet, refusalEnvelope)
		rg2, rerr2 := CompileNativeObligationGraph(packet, refusalEnvelope)
		if rerr1 == nil || rerr2 == nil {
			t.Fatal("expected refusal on unsupported backend")
		}
		var gref1, gref2 *Refusal
		if !errors.As(rerr1, &gref1) || !errors.As(rerr2, &gref2) {
			t.Fatalf("expected *Refusal, got %T, %T", rerr1, rerr2)
		}
		if !reflect.DeepEqual(gref1, gref2) {
			t.Fatalf("graph refusals are not deeply equal:\ngref1 = %+v\ngref2 = %+v", gref1, gref2)
		}
		if !reflect.DeepEqual(rg1, rg2) {
			t.Fatal("graphs on refusal are not deeply equal")
		}
	})

	t.Run("ReplayCorpus", func(t *testing.T) {
		corpusRaw := replayFixture(t, "corpus.json")
		corpus, err := ParseReplayCorpus(corpusRaw)
		if err != nil {
			t.Fatal(err)
		}

		firstLedger, err1 := Replay(corpus)
		if err1 != nil {
			t.Fatalf("first Replay failed: %v", err1)
		}
		secondLedger, err2 := Replay(corpus)
		if err2 != nil {
			t.Fatalf("second Replay failed: %v", err2)
		}
		if !reflect.DeepEqual(firstLedger, secondLedger) {
			t.Fatalf("Replay ledgers are not deeply equal:\nfirst:  %+v\nsecond: %+v", firstLedger, secondLedger)
		}

		firstBytes, err1 := MarshalReplayLedger(firstLedger)
		if err1 != nil {
			t.Fatal(err1)
		}
		secondBytes, err2 := MarshalReplayLedger(secondLedger)
		if err2 != nil {
			t.Fatal(err2)
		}
		if !bytes.Equal(firstBytes, secondBytes) {
			t.Fatal("MarshalReplayLedger bytes are not identical")
		}
	})

	t.Run("Scaffold", func(t *testing.T) {
		scaffolds := []Scaffold{
			{Family: "qwen38", Topology: "prenorm"},
			{Family: "deepseek2", Topology: "postnorm"},
			{Family: "kimi", Topology: "parallel"},
			{Family: "generic", Topology: "identity"},
		}
		for _, sc := range scaffolds {
			res1, err1 := Run(sc)
			if err1 != nil {
				t.Fatalf("first Run failed for %+v: %v", sc, err1)
			}
			res2, err2 := Run(sc)
			if err2 != nil {
				t.Fatalf("second Run failed for %+v: %v", sc, err2)
			}
			if !reflect.DeepEqual(res1, res2) {
				t.Fatalf("Run scaffold outputs are not deeply equal for %+v:\nres1: %+v\nres2: %+v", sc, res1, res2)
			}
		}
	})

	t.Run("ConcurrentRaceWitness", func(t *testing.T) {
		const goroutines = 16
		const iterations = 8
		var wg sync.WaitGroup
		wg.Add(goroutines)

		errCh := make(chan error, goroutines*iterations)

		unknownDeltaRaw := fixture(t, "qwen38-unknown-delta.json")

		for g := 0; g < goroutines; g++ {
			go func() {
				defer wg.Done()
				for i := 0; i < iterations; i++ {
					// 1. Valid intake
					p1, e1 := CompileManifest(validRaw)
					p2, e2 := CompileManifest(validRaw)
					if e1 != nil || e2 != nil {
						errCh <- fmt.Errorf("concurrent CompileManifest error: %v, %v", e1, e2)
						return
					}
					if !reflect.DeepEqual(p1, p2) {
						errCh <- fmt.Errorf("concurrent CompileManifest outputs are not deeply equal")
						return
					}

					// 2. Refusal intake
					_, re1 := CompileManifest(unknownDeltaRaw)
					_, re2 := CompileManifest(unknownDeltaRaw)
					var rf1, rf2 *Refusal
					if !errors.As(re1, &rf1) || !errors.As(re2, &rf2) {
						errCh <- fmt.Errorf("concurrent refusal conversion failed")
						return
					}
					if !reflect.DeepEqual(rf1, rf2) {
						errCh <- fmt.Errorf("concurrent refusals are not deeply equal: %+v vs %+v", rf1, rf2)
						return
					}

					// 3. Obligation graph
					g1, ge1 := CompileNativeObligationGraph(p1, envelope)
					g2, ge2 := CompileNativeObligationGraph(p2, envelope)
					if ge1 != nil || ge2 != nil {
						errCh <- fmt.Errorf("concurrent CompileNativeObligationGraph error: %v, %v", ge1, ge2)
						return
					}
					if !reflect.DeepEqual(g1, g2) {
						errCh <- fmt.Errorf("concurrent CompileNativeObligationGraph outputs are not deeply equal")
						return
					}

					// 4. Scaffold
					s1, se1 := Run(Scaffold{Family: "qwen38", Topology: "prenorm"})
					s2, se2 := Run(Scaffold{Family: "qwen38", Topology: "prenorm"})
					if se1 != nil || se2 != nil {
						errCh <- fmt.Errorf("concurrent Run error: %v, %v", se1, se2)
						return
					}
					if !reflect.DeepEqual(s1, s2) {
						errCh <- fmt.Errorf("concurrent Run outputs are not deeply equal")
						return
					}
				}
			}()
		}

		wg.Wait()
		close(errCh)
		for err := range errCh {
			t.Fatal(err)
		}
	})
}
