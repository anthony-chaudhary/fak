package quantmeta

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// contract_test.go is the #6222 witness for the neutral quantization descriptor
// spine. It proves the two properties the issue's acceptance gate names:
//
//   - STABLE DESCRIPTORS — every fixture parses into typed Go values and encodes
//     back to a byte-identical canonical golden, and re-encoding the golden is a
//     fixed point (idempotence). A descriptor that round-trips is one a peer
//     runtime can hand to fak and get back unchanged.
//   - UNKNOWN-FIELD BEHAVIOR — fields fak does not know are PRESERVED, not
//     dropped and not fatal. This is the neutrality property that keeps fak from
//     becoming the format owner: a producer's private key survives the round trip
//     untouched, so passing a descriptor through fak is never lossy.
//
// Run with -update to regenerate the .golden.json canonical forms.

var update = flag.Bool("update", false, "regenerate the canonical .golden.json fixtures")

// fixtures are the hand-authored, producer-shaped inputs. They deliberately span
// the descriptor axes #6222 enumerates -- weight, activation, KV, scale,
// zero-point, grouping, codebook, sparsity and training provenance -- across
// unrelated industry families so no single family is privileged by the fixture
// set itself.
var fixtures = []string{
	"gguf_q4k",
	"gptq_int4",
	"fp8_weight_activation",
	"bitnet_ternary",
	"kv_int4_sparse",
	"codebook_delegate",
	"future_schema",
	"unknown_everywhere",
}

func readFixture(t *testing.T, name string) []byte {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatalf("read fixture %s: %v", name, err)
	}
	return b
}

func TestGoldenRoundTrip(t *testing.T) {
	for _, name := range fixtures {
		t.Run(name, func(t *testing.T) {
			in := readFixture(t, name+".input.json")
			d, err := Parse(in)
			if err != nil {
				t.Fatalf("Parse(%s.input.json) = %v, want no error", name, err)
			}
			got, err := Encode(d)
			if err != nil {
				t.Fatalf("Encode: %v", err)
			}
			goldenPath := filepath.Join("testdata", name+".golden.json")
			if *update {
				if err := os.WriteFile(goldenPath, got, 0o644); err != nil {
					t.Fatalf("update golden: %v", err)
				}
			}
			want, err := os.ReadFile(goldenPath)
			if err != nil {
				t.Fatalf("read golden (re-run with -update to create it): %v", err)
			}
			if string(got) != string(want) {
				t.Errorf("canonical encoding drifted from golden %s\n got: %s\nwant: %s", goldenPath, got, want)
			}
			// Idempotence: the canonical form is a fixed point. A descriptor that
			// re-encodes differently is not stable enough to hand between runtimes.
			d2, err := Parse(want)
			if err != nil {
				t.Fatalf("Parse(golden) = %v, want no error", err)
			}
			got2, err := Encode(d2)
			if err != nil {
				t.Fatalf("Encode(Parse(golden)): %v", err)
			}
			if string(got2) != string(want) {
				t.Errorf("canonical form is not a fixed point\n got: %s\nwant: %s", got2, want)
			}
		})
	}
}

// TestUnknownFieldsPreserved is the neutrality witness at the codec layer: a key
// fak has never heard of survives parse+encode byte-for-byte, at BOTH the
// descriptor level and inside a nested tensor descriptor. Dropping it would
// silently make fak the authority on which fields are legal.
func TestUnknownFieldsPreserved(t *testing.T) {
	t.Run("descriptor level", func(t *testing.T) {
		d, err := Parse(readFixture(t, "gptq_int4.input.json"))
		if err != nil {
			t.Fatalf("Parse: %v", err)
		}
		raw, ok := d.Extra["desc_act"]
		if !ok {
			t.Fatalf("unknown descriptor key %q was dropped; Extra = %v", "desc_act", d.Extra)
		}
		if strings.TrimSpace(string(raw)) != "true" {
			t.Errorf("Extra[desc_act] = %s, want true", raw)
		}
		out, err := Encode(d)
		if err != nil {
			t.Fatalf("Encode: %v", err)
		}
		if !strings.Contains(string(out), `"desc_act"`) {
			t.Errorf("unknown key absent from re-encoded output:\n%s", out)
		}
	})

	t.Run("nested tensor level", func(t *testing.T) {
		d, err := Parse(readFixture(t, "gguf_q4k.input.json"))
		if err != nil {
			t.Fatalf("Parse: %v", err)
		}
		if d.Weight == nil {
			t.Fatal("Weight = nil, want a parsed tensor descriptor")
		}
		raw, ok := d.Weight.Extra["llama_cpp_type"]
		if !ok {
			t.Fatalf("unknown tensor key %q was dropped; Extra = %v", "llama_cpp_type", d.Weight.Extra)
		}
		var got string
		if err := json.Unmarshal(raw, &got); err != nil {
			t.Fatalf("unmarshal preserved key: %v", err)
		}
		if got != "Q4_K_M" {
			t.Errorf("Weight.Extra[llama_cpp_type] = %q, want %q", got, "Q4_K_M")
		}
	})

	t.Run("deeply nested unknown value", func(t *testing.T) {
		d, err := Parse(readFixture(t, "future_schema.input.json"))
		if err != nil {
			t.Fatalf("Parse: %v", err)
		}
		raw, ok := d.Extra["future_only_field"]
		if !ok {
			t.Fatalf("unknown structured key was dropped; Extra = %v", d.Extra)
		}
		var nested struct {
			Nested []json.RawMessage `json:"nested"`
		}
		if err := json.Unmarshal(raw, &nested); err != nil {
			t.Fatalf("preserved value is not intact JSON: %v", err)
		}
		if len(nested.Nested) != 3 {
			t.Errorf("nested array len = %d, want 3 (structure was flattened)", len(nested.Nested))
		}
	})
}

// TestParsedFieldsAreTyped guards against a codec that "round-trips" by keeping
// everything in Extra and typing nothing. Each axis #6222 names must land in a
// real Go field.
func TestParsedFieldsAreTyped(t *testing.T) {
	d, err := Parse(readFixture(t, "kv_int4_sparse.input.json"))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if d.Weight == nil || d.Weight.Format != FormatInt8 {
		t.Errorf("Weight.Format = %v, want %v", d.Weight, FormatInt8)
	}
	if d.KV == nil || d.KV.Key == nil || d.KV.Value == nil {
		t.Fatalf("KV = %+v, want both key and value descriptors", d.KV)
	}
	if d.KV.Key.Format != FormatInt4 {
		t.Errorf("KV.Key.Format = %q, want %q", d.KV.Key.Format, FormatInt4)
	}
	if d.KV.Key.ZeroPoint == nil || !d.KV.Key.ZeroPoint.Present {
		t.Errorf("KV.Key.ZeroPoint = %+v, want a present zero point", d.KV.Key.ZeroPoint)
	}
	if d.KV.Key.Scale == nil || d.KV.Key.Scale.Granularity != GranularityPerHead {
		t.Errorf("KV.Key.Scale = %+v, want per-head granularity", d.KV.Key.Scale)
	}
	if d.KV.SinkTokensFullPrecision != 4 {
		t.Errorf("KV.SinkTokensFullPrecision = %d, want 4", d.KV.SinkTokensFullPrecision)
	}
	if d.Sparsity == nil || d.Sparsity.N != 2 || d.Sparsity.M != 4 {
		t.Errorf("Sparsity = %+v, want the 2:4 structured pattern", d.Sparsity)
	}
	if d.Provenance.TrainingStage != TrainingStagePostTraining {
		t.Errorf("Provenance.TrainingStage = %q, want %q", d.Provenance.TrainingStage, TrainingStagePostTraining)
	}
	if d.Provenance.CalibrationDisclosed == nil || !*d.Provenance.CalibrationDisclosed {
		t.Errorf("Provenance.CalibrationDisclosed = %v, want an explicit true", d.Provenance.CalibrationDisclosed)
	}
	if d.Provenance.ProducerID != "example-toolchain" {
		t.Errorf("Provenance.ProducerID = %q, want %q", d.Provenance.ProducerID, "example-toolchain")
	}
	if d.Artifact == nil || d.Artifact.ContainerID != "safetensors" {
		t.Errorf("Artifact = %+v, want a typed safetensors container", d.Artifact)
	}
	if d.Weight.Bits != 8 {
		t.Errorf("Weight.Bits = %v, want 8 (bits_per_value must land in the typed field)", d.Weight.Bits)
	}

	cb, err := Parse(readFixture(t, "codebook_delegate.input.json"))
	if err != nil {
		t.Fatalf("Parse codebook: %v", err)
	}
	if cb.Weight == nil || cb.Weight.Codebook == nil {
		t.Fatalf("Weight.Codebook = nil, want a typed codebook descriptor")
	}
	if cb.Weight.Codebook.Entries != 256 || cb.Weight.Codebook.ResidualStages != 2 {
		t.Errorf("Codebook = %+v, want 256 entries and 2 residual stages", cb.Weight.Codebook)
	}

	bn, err := Parse(readFixture(t, "bitnet_ternary.input.json"))
	if err != nil {
		t.Fatalf("Parse ternary: %v", err)
	}
	if bn.Weight.Format != FormatTernary {
		t.Errorf("Weight.Format = %q, want %q", bn.Weight.Format, FormatTernary)
	}
	if bn.Provenance.TrainingStage != TrainingStageTrainedNatively {
		t.Errorf("TrainingStage = %q, want %q", bn.Provenance.TrainingStage, TrainingStageTrainedNatively)
	}
	if bn.Activation == nil || bn.Activation.Format != FormatInt8 {
		t.Errorf("Activation = %+v, want an int8 activation descriptor", bn.Activation)
	}
}

// TestNoFieldLoss is the anti-vacuity witness for "stable descriptors". A golden
// file regenerated from a lossy encoder certifies the loss instead of catching
// it, so stability is asserted here against the HAND-AUTHORED producer input
// rather than against our own output: parse-then-encode must be semantically
// identity, key for key, at every depth.
//
// This is the neutrality property stated precisely. fak is not the owner of this
// format; a descriptor that loses a producer's field on the way through has
// quietly made fak the authority on which fields are allowed to exist.
func TestNoFieldLoss(t *testing.T) {
	for _, name := range fixtures {
		t.Run(name, func(t *testing.T) {
			in := readFixture(t, name+".input.json")
			d, err := Parse(in)
			if err != nil {
				t.Fatalf("Parse: %v", err)
			}
			out, err := Encode(d)
			if err != nil {
				t.Fatalf("Encode: %v", err)
			}
			var want, got any
			if err := json.Unmarshal(in, &want); err != nil {
				t.Fatalf("unmarshal input: %v", err)
			}
			if err := json.Unmarshal(out, &got); err != nil {
				t.Fatalf("unmarshal output: %v", err)
			}
			for _, path := range missingPaths(t, "", want, got) {
				t.Errorf("round trip lost or changed %s", path)
			}
		})
	}
}

// missingPaths reports every key path present in want that is absent from got or
// carries a different value there.
func missingPaths(t *testing.T, prefix string, want, got any) []string {
	t.Helper()
	switch w := want.(type) {
	case map[string]any:
		g, ok := got.(map[string]any)
		if !ok {
			return []string{prefix + " (object became " + describeJSON(got) + ")"}
		}
		var out []string
		keys := make([]string, 0, len(w))
		for k := range w {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			gv, present := g[k]
			path := prefix + "." + k
			if !present {
				out = append(out, path+" (dropped)")
				continue
			}
			out = append(out, missingPaths(t, path, w[k], gv)...)
		}
		return out
	case []any:
		g, ok := got.([]any)
		if !ok || len(g) != len(w) {
			return []string{prefix + " (array changed)"}
		}
		var out []string
		for i := range w {
			out = append(out, missingPaths(t, prefix+"["+strconv.Itoa(i)+"]", w[i], g[i])...)
		}
		return out
	default:
		if !reflect.DeepEqual(want, got) {
			return []string{fmt.Sprintf("%s (%v -> %v)", prefix, want, got)}
		}
		return nil
	}
}

func describeJSON(v any) string { return fmt.Sprintf("%T", v) }

// TestExtraCollisionRefused: an Extra key that shadows a known field would emit
// duplicate JSON keys, which is a silently ambiguous document. Encode refuses it
// with a typed error rather than emitting it.
func TestExtraCollisionRefused(t *testing.T) {
	d := Descriptor{
		Schema: SchemaVersion,
		Extra:  map[string]json.RawMessage{"weight": json.RawMessage(`{"format":"int4"}`)},
	}
	if _, err := Encode(d); err == nil {
		t.Fatal("Encode() = nil error, want a refusal for an Extra key shadowing a known field")
	}
}

func TestParseRejectsMalformedJSON(t *testing.T) {
	for _, bad := range []string{``, `{`, `[]`, `"a string"`, `{"schema":`} {
		if _, err := Parse([]byte(bad)); err == nil {
			t.Errorf("Parse(%q) = nil error, want a parse error", bad)
		}
	}
}
