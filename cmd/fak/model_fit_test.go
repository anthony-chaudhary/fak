package main

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/fitplan"
)

const modelFitFixture = `{"pools":[{"id":"small","usable_bytes":16},{"id":"large","usable_bytes":32}],"placement":{"name":"unequal","support":"supported","support_evidence":"caller-declared backend and links","weights":{"replicated_bytes":8,"sharded_bytes":24,"shard_count":8},"ranks":[{"rank":0,"pool_id":"small","shards":2,"kv_bytes":1,"workspace_bytes":1,"extra_bytes":0},{"rank":1,"pool_id":"large","shards":6,"kv_bytes":1,"workspace_bytes":1,"extra_bytes":0}]}}`

// fak-test:runtime fast est=10ms
// fak-test:justify why=Expose modeled fit and check refusal through the CLI artifact contract. when=changed:cmd/fak/model_fit*
func TestModelFitCLIPlanAndCheck(t *testing.T) {
	for _, tc := range []struct {
		name, input string
		code        int
		eligible    bool
	}{
		{"fits", modelFitFixture, 0, true},
		{"rank-overflow", strings.Replace(modelFitFixture, `"usable_bytes":16`, `"usable_bytes":15`, 1), 1, false},
		{"unknown-capacity", strings.Replace(modelFitFixture, `"usable_bytes":16`, `"usable_bytes":0`, 1), 1, false},
		{"unknown-support", strings.Replace(modelFitFixture, `"support":"supported"`, `"support":"unknown"`, 1), 1, false},
		{"missing-evidence", strings.Replace(modelFitFixture, `"caller-declared backend and links"`, `""`, 1), 1, false},
		{"unsupported", strings.Replace(modelFitFixture, `"support":"supported"`, `"support":"unsupported"`, 1), 1, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, mode := range []string{"plan", "check"} {
				var out, errOut bytes.Buffer
				code := runModelFit(strings.NewReader(tc.input), &out, &errOut, []string{mode, "--input", "-", "--json"})
				want := tc.code
				if mode == "plan" {
					want = 0
				}
				if code != want {
					t.Fatalf("mode=%s code=%d want=%d stderr=%s", mode, code, want, errOut.String())
				}
				var artifact modelFitArtifact
				if err := json.Unmarshal(out.Bytes(), &artifact); err != nil {
					t.Fatal(err)
				}
				if artifact.Schema != "fak-model-fit/1" || !artifact.ModeledOnly || artifact.Qualification != "unverified" || len(artifact.Reports) != 1 || artifact.Reports[0].Eligible != tc.eligible {
					t.Fatalf("unexpected artifact: %+v", artifact)
				}
			}
		})
	}
}

// fak-test:runtime fast est=20ms
// fak-test:justify why=Reject malformed or oversized declarations before producing a fit artifact. when=changed:cmd/fak/model_fit*
func TestModelFitCLIInputRefusals(t *testing.T) {
	for _, input := range []string{`{}`, `null`, modelFitFixture + ` {}`, strings.Replace(modelFitFixture, `"shards":2`, `"shards":2,"typo":1`, 1), strings.Replace(modelFitFixture, `"kv_bytes":1,`, ``, 1), strings.Replace(modelFitFixture, `"kv_bytes":1`, `"kv_bytes":null`, 1), strings.Replace(modelFitFixture, `"replicated_bytes":8,`, ``, 1), strings.Repeat(" ", modelFitInputLimit+1)} {
		var out, errOut bytes.Buffer
		if code := runModelFit(strings.NewReader(input), &out, &errOut, []string{"plan", "--input", "-"}); code != 2 || out.Len() != 0 {
			t.Fatalf("code=%d output=%s", code, out.String())
		}
	}
}

// fak-test:runtime fast est=10ms
// fak-test:justify why=Comparison preserves alternative identity and refuses changing the model. when=changed:cmd/fak/model_fit*
func TestModelFitCLICompareFixedWeights(t *testing.T) {
	var base map[string]json.RawMessage
	if err := json.Unmarshal([]byte(modelFitFixture), &base); err != nil {
		t.Fatal(err)
	}
	placement := string(base["placement"])
	second := strings.Replace(placement, `"unequal"`, `"equal"`, 1)
	second = strings.ReplaceAll(second, `"shards":2`, `"shards":4`)
	second = strings.ReplaceAll(second, `"shards":6`, `"shards":4`)
	input := `{"pools":` + string(base["pools"]) + `,"placements":[` + placement + `,` + second + `]}`
	var out, errOut bytes.Buffer
	if code := runModelFit(strings.NewReader(input), &out, &errOut, []string{"compare", "--input", "-"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	var artifact modelFitArtifact
	if err := json.Unmarshal(out.Bytes(), &artifact); err != nil {
		t.Fatal(err)
	}
	if len(artifact.Reports) != 2 || artifact.Reports[0].Name != "unequal" || artifact.Reports[1].Name != "equal" || !artifact.Reports[0].Eligible || artifact.Reports[1].Eligible {
		t.Fatalf("reports=%+v", artifact.Reports)
	}

	second = strings.Replace(second, `"sharded_bytes":24`, `"sharded_bytes":25`, 1)
	input = `{"pools":` + string(base["pools"]) + `,"placements":[` + placement + `,` + second + `]}`
	if code := runModelFit(strings.NewReader(input), io.Discard, io.Discard, []string{"compare", "--input", "-"}); code != 2 {
		t.Fatalf("changed model code=%d", code)
	}
}

func modelFitFixtureFile(t *testing.T, name string) string {
	t.Helper()
	raw, err := os.ReadFile("../../internal/fitplan/testdata/" + name)
	if err != nil {
		t.Fatal(err)
	}
	return string(raw)
}

// fak-test:runtime fast est=10ms
// fak-test:justify why=Pin the tiered-refusal artifact through the CLI: check refuses with byte-proven next_tiers, eligible reports omit the key, exit codes stay unchanged. when=changed:cmd/fak/model_fit*,internal/fitplan/next_tiers*
func TestModelFitCLINextTiers(t *testing.T) {
	t.Run("too-small-check-refuses-with-multi-device", func(t *testing.T) {
		input := modelFitFixtureFile(t, "ex-too-small.json")
		for _, tc := range []struct {
			mode string
			code int
		}{
			{"check", 1},
			{"plan", 0},
		} {
			var out, errOut bytes.Buffer
			if code := runModelFit(strings.NewReader(input), &out, &errOut, []string{tc.mode, "--input", "-"}); code != tc.code {
				t.Fatalf("mode=%s code=%d want=%d stderr=%s", tc.mode, code, tc.code, errOut.String())
			}
			var artifact modelFitArtifact
			if err := json.Unmarshal(out.Bytes(), &artifact); err != nil {
				t.Fatal(err)
			}
			rep := artifact.Reports[0]
			if rep.CapacityStatus != fitplan.StatusRefused || rep.Eligible {
				t.Fatalf("report=%+v", rep)
			}
			if len(rep.NextTiers) != 1 || rep.NextTiers[0].Tier != fitplan.NextTierMultiDevice || rep.NextTiers[0].ShortfallBytes != 38654705664 {
				t.Fatalf("next_tiers=%+v", rep.NextTiers)
			}
		}
	})
	t.Run("host-pool-names-host-offload-first", func(t *testing.T) {
		var out, errOut bytes.Buffer
		if code := runModelFit(strings.NewReader(modelFitFixtureFile(t, "ex-host-offload.json")), &out, &errOut, []string{"check", "--input", "-"}); code != 1 {
			t.Fatalf("code=%d stderr=%s", code, errOut.String())
		}
		var artifact modelFitArtifact
		if err := json.Unmarshal(out.Bytes(), &artifact); err != nil {
			t.Fatal(err)
		}
		rep := artifact.Reports[0]
		if len(rep.NextTiers) != 2 || rep.NextTiers[0].Tier != fitplan.NextTierHostOffload || rep.NextTiers[0].ShortfallBytes != 12884901888 || rep.NextTiers[1].Tier != fitplan.NextTierMultiDevice || rep.NextTiers[1].ShortfallBytes != 12884901888 {
			t.Fatalf("next_tiers=%+v", rep.NextTiers)
		}
	})
	t.Run("unknown-capacity-closes-with-reason-and-no-tiers", func(t *testing.T) {
		var out, errOut bytes.Buffer
		if code := runModelFit(strings.NewReader(modelFitFixtureFile(t, "ex-unknown-capacity.json")), &out, &errOut, []string{"check", "--input", "-"}); code != 1 {
			t.Fatalf("code=%d stderr=%s", code, errOut.String())
		}
		var artifact modelFitArtifact
		if err := json.Unmarshal(out.Bytes(), &artifact); err != nil {
			t.Fatal(err)
		}
		rep := artifact.Reports[0]
		closed := false
		for _, reason := range rep.Reasons {
			if reason == fitplan.PlacementPoolUnknown {
				closed = true
			}
		}
		if !closed || rep.NextTiers != nil || strings.Contains(out.String(), "next_tiers") {
			t.Fatalf("report=%+v output=%s", rep, out.String())
		}
	})
	for _, fixture := range []string{"dense-16g.json", "ex-asym-supported.json", "ex-asym.json"} {
		t.Run("eligible-or-support-only-omits-next-tiers:"+fixture, func(t *testing.T) {
			var out, errOut bytes.Buffer
			want := 0
			if fixture == "ex-asym.json" {
				want = 1 // capacity fits, support unknown: check still refuses
			}
			if code := runModelFit(strings.NewReader(modelFitFixtureFile(t, fixture)), &out, &errOut, []string{"check", "--input", "-"}); code != want {
				t.Fatalf("code=%d want=%d stderr=%s", code, want, errOut.String())
			}
			if strings.Contains(out.String(), "next_tiers") {
				t.Fatalf("output=%s", out.String())
			}
		})
	}
}

// fak-test:runtime fast est=10ms
// fak-test:justify why=Compose produces a fixed-model declaration consumable by check while refusing unknown budgets and pins. when=changed:cmd/fak/model_fit*
func TestModelFitCLIComposeRoundTrip(t *testing.T) {
	input := strings.ReplaceAll(strings.ReplaceAll(modelFitFixture, `"shards":2`, `"shards":0`), `"shards":6`, `"shards":0`)
	var composed, errOut bytes.Buffer
	if code := runModelFit(strings.NewReader(input), &composed, &errOut, []string{"compose", "--input", "-"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	var out bytes.Buffer
	if code := runModelFit(bytes.NewReader(composed.Bytes()), &out, &errOut, []string{"check", "--input", "-"}); code != 0 {
		t.Fatalf("roundtrip code=%d stderr=%s", code, errOut.String())
	}
	var artifact modelFitArtifact
	if err := json.Unmarshal(out.Bytes(), &artifact); err != nil {
		t.Fatal(err)
	}
	if len(artifact.Reports) != 1 || !artifact.Reports[0].Eligible || artifact.Reports[0].Ranks[0].ShardedBytes != 6 || artifact.Reports[0].Ranks[1].ShardedBytes != 18 {
		t.Fatalf("artifact=%+v", artifact)
	}
	for _, tc := range []struct {
		input string
		code  int
	}{
		{modelFitFixture, 2},
		{strings.Replace(input, `"usable_bytes":16`, `"usable_bytes":0`, 1), 1},
		{strings.Replace(input, `"usable_bytes":32`, `"usable_bytes":16`, 1), 1},
	} {
		var output bytes.Buffer
		if code := runModelFit(strings.NewReader(tc.input), &output, io.Discard, []string{"compose", "--input", "-"}); code != tc.code || output.Len() != 0 {
			t.Fatalf("code=%d want=%d output=%s", code, tc.code, output.String())
		}
	}
}
