// workload_plan.go ports InferenceX's canonical-JSON workload-plan identity to Go.
//
// NOTICE: ported from SemiAnalysisAI/InferenceX, video-generation/specs/video_spec.py
// (canonical_json_bytes, plan_for, plan_sha256), pinned at commit
// sorted-key, compact, UTF-8 JSON that refuses NaN/Inf, and renames a plan whose
// repetition/warmup counts are overridden. This port gives fak benchmark receipts
// the same workload identity: content is hashed, host/GPU/port never are.
package perfledger

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"
)

// WorkloadCase is one resolved benchmark request: a stable id, the prompt text
// actually sent, and the sampling seed.
type WorkloadCase struct {
	CaseID string
	Prompt string
	Seed   int64
}

// WorkloadPlan is the executable workload for one benchmark run. Every field
// above the runtime block feeds WorkloadPlanSHA256; Host, GPU and Port describe
// where the plan ran and are deliberately excluded so the same workload keeps
// one identity across machines.
type WorkloadPlan struct {
	PlanID        string
	ModelID       string
	ModelRevision string
	Cases         []WorkloadCase
	Repetitions   int
	WarmupRuns    int
	Generation    map[string]any
	// Fairness is optional; a nil map is omitted from the identity entirely.
	Fairness map[string]any

	Host string
	GPU  string
	Port int
}

// WorkloadPlanIdentity returns the JSON-shaped document that the plan hash
// covers, keyed as the upstream plan object minus its video-only case flags.
// A fairness block marks a schema 2 plan, which upstream stamps with
// schema_version; schema 1 plans omit the field to keep their hashes.
func WorkloadPlanIdentity(p WorkloadPlan) map[string]any {
	cases := make([]any, len(p.Cases))
	for i, c := range p.Cases {
		cases[i] = map[string]any{"case_id": c.CaseID, "prompt": c.Prompt, "seed": c.Seed}
	}
	generation := p.Generation
	if generation == nil {
		generation = map[string]any{}
	}
	doc := map[string]any{
		"plan_id":        p.PlanID,
		"model_id":       p.ModelID,
		"model_revision": p.ModelRevision,
		"generation":     generation,
		"cases":          cases,
		"repetitions":    p.Repetitions,
		"warmup_runs":    p.WarmupRuns,
	}
	if p.Fairness != nil {
		doc["fairness"] = p.Fairness
		doc["schema_version"] = 2
	}
	return doc
}

// WorkloadPlanSHA256 is the lowercase hex plan_sha256 of the plan's identity.
func WorkloadPlanSHA256(p WorkloadPlan) (string, error) {
	b, err := CanonicalPlanJSON(WorkloadPlanIdentity(p))
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:]), nil
}

// WithWorkloadCountOverride returns a copy of p running repetitions/warmupRuns.
// When either count differs from p's own, the plan id gains
// "-r<repetitions>-w<warmupRuns>" so a shortened run never shares the full
// run's identity; passing p's own counts returns an identical plan.
func WithWorkloadCountOverride(p WorkloadPlan, repetitions, warmupRuns int) (WorkloadPlan, error) {
	if repetitions < 1 {
		return WorkloadPlan{}, fmt.Errorf("perfledger: repetitions must be >= 1, got %d", repetitions)
	}
	if warmupRuns < 0 {
		return WorkloadPlan{}, fmt.Errorf("perfledger: warmup_runs must be >= 0, got %d", warmupRuns)
	}
	out := p
	out.Cases = append([]WorkloadCase(nil), p.Cases...)
	if repetitions != p.Repetitions || warmupRuns != p.WarmupRuns {
		out.PlanID = fmt.Sprintf("%s-r%d-w%d", p.PlanID, repetitions, warmupRuns)
	}
	out.Repetitions = repetitions
	out.WarmupRuns = warmupRuns
	return out, nil
}

// CanonicalPlanJSON encodes v as sorted-key, compact, UTF-8 JSON without HTML
// or non-ASCII escaping, refusing NaN, ±Inf and invalid UTF-8. Maps, slices,
// strings, bools, integers and floats are encoded directly, with floats in
// Python repr form so bytes match the upstream hashes; other values (structs)
// go through encoding/json first and keep its number text.
func CanonicalPlanJSON(v any) ([]byte, error) {
	var buf bytes.Buffer
	if err := wplanEncode(&buf, reflect.ValueOf(v)); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

var wplanNumberType = reflect.TypeOf(json.Number(""))

func wplanEncode(buf *bytes.Buffer, v reflect.Value) error {
	if !v.IsValid() {
		buf.WriteString("null")
		return nil
	}
	if v.Type() == wplanNumberType {
		s := v.String()
		if !json.Valid([]byte(s)) {
			return fmt.Errorf("perfledger: invalid json.Number %q", s)
		}
		buf.WriteString(s)
		return nil
	}
	switch v.Kind() {
	case reflect.Interface, reflect.Pointer:
		if v.IsNil() {
			buf.WriteString("null")
			return nil
		}
		return wplanEncode(buf, v.Elem())
	case reflect.Bool:
		buf.WriteString(strconv.FormatBool(v.Bool()))
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		buf.WriteString(strconv.FormatInt(v.Int(), 10))
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		buf.WriteString(strconv.FormatUint(v.Uint(), 10))
	case reflect.Float32, reflect.Float64:
		s, err := wplanFormatFloat(v.Float(), v.Type().Bits())
		if err != nil {
			return err
		}
		buf.WriteString(s)
	case reflect.String:
		return wplanEncodeString(buf, v.String())
	case reflect.Slice, reflect.Array:
		if v.Kind() == reflect.Slice && v.IsNil() {
			buf.WriteString("null")
			return nil
		}
		buf.WriteByte('[')
		for i := 0; i < v.Len(); i++ {
			if i > 0 {
				buf.WriteByte(',')
			}
			if err := wplanEncode(buf, v.Index(i)); err != nil {
				return err
			}
		}
		buf.WriteByte(']')
	case reflect.Map:
		if v.Type().Key().Kind() != reflect.String {
			return fmt.Errorf("perfledger: canonical JSON needs string map keys, got %s", v.Type().Key())
		}
		if v.IsNil() {
			buf.WriteString("null")
			return nil
		}
		keys := make([]string, 0, v.Len())
		for _, k := range v.MapKeys() {
			keys = append(keys, k.String())
		}
		// Byte order equals code-point order for valid UTF-8, matching Python's sort_keys.
		sort.Strings(keys)
		buf.WriteByte('{')
		for i, k := range keys {
			if i > 0 {
				buf.WriteByte(',')
			}
			if err := wplanEncodeString(buf, k); err != nil {
				return err
			}
			buf.WriteByte(':')
			if err := wplanEncode(buf, v.MapIndex(reflect.ValueOf(k).Convert(v.Type().Key()))); err != nil {
				return err
			}
		}
		buf.WriteByte('}')
	default:
		return wplanEncodeViaJSON(buf, v)
	}
	return nil
}

// wplanEncodeViaJSON normalizes values with custom JSON shapes (struct tags,
// Marshalers) into a generic tree, then re-encodes it canonically.
func wplanEncodeViaJSON(buf *bytes.Buffer, v reflect.Value) error {
	raw, err := json.Marshal(v.Interface())
	if err != nil {
		return fmt.Errorf("perfledger: canonical JSON: %w", err)
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	var tree any
	if err := dec.Decode(&tree); err != nil {
		return fmt.Errorf("perfledger: canonical JSON: %w", err)
	}
	return wplanEncode(buf, reflect.ValueOf(tree))
}

// wplanFormatFloat renders f as Python's float repr: shortest round-trip
// digits, fixed notation for decimal exponents in [-4, 16) with a mandatory
// fractional part, scientific otherwise.
func wplanFormatFloat(f float64, bits int) (string, error) {
	if math.IsNaN(f) || math.IsInf(f, 0) {
		return "", fmt.Errorf("perfledger: canonical JSON refuses non-finite float %v", f)
	}
	sci := strconv.FormatFloat(f, 'e', -1, bits)
	exp := 0
	if i := strings.IndexByte(sci, 'e'); i >= 0 {
		exp, _ = strconv.Atoi(sci[i+1:])
	}
	if exp < -4 || exp >= 16 {
		return sci, nil
	}
	s := strconv.FormatFloat(f, 'f', -1, bits)
	if strings.IndexByte(s, '.') < 0 {
		s += ".0"
	}
	return s, nil
}

func wplanEncodeString(buf *bytes.Buffer, s string) error {
	if !utf8.ValidString(s) {
		return fmt.Errorf("perfledger: canonical JSON refuses invalid UTF-8 %q", s)
	}
	const hexDigits = "0123456789abcdef"
	buf.WriteByte('"')
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch c {
		case '"':
			buf.WriteString(`\"`)
		case '\\':
			buf.WriteString(`\\`)
		case '\n':
			buf.WriteString(`\n`)
		case '\r':
			buf.WriteString(`\r`)
		case '\t':
			buf.WriteString(`\t`)
		case '\b':
			buf.WriteString(`\b`)
		case '\f':
			buf.WriteString(`\f`)
		default:
			if c < 0x20 {
				buf.WriteString(`\u00`)
				buf.WriteByte(hexDigits[c>>4])
				buf.WriteByte(hexDigits[c&0xf])
			} else {
				buf.WriteByte(c)
			}
		}
	}
	buf.WriteByte('"')
	return nil
}
