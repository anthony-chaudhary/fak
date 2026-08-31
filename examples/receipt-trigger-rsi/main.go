package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"sort"
	"strings"
	"time"
)

const (
	schemaDecision = "fak-receipt-trigger-decision/1"
	defaultNow     = "2026-08-31T12:00:00Z"
)

type Receipt struct {
	Schema        string   `json:"schema"`
	Reason        string   `json:"reason,omitempty"`
	Producer      string   `json:"producer"`
	ProducedAt    string   `json:"produced_at"`
	EffectKey     string   `json:"effect_key"`
	Recursion     bool     `json:"recursion,omitempty"`
	Capacity      int      `json:"capacity"`
	ExpectedValue int      `json:"expected_value"`
	SampleCount   int      `json:"sample_count,omitempty"`
	EvidenceRefs  []string `json:"evidence_refs,omitempty"`
}

type Decision struct {
	Schema       string   `json:"schema"`
	Decision     string   `json:"decision"`
	Reason       string   `json:"reason"`
	Consumer     string   `json:"consumer"`
	Authority    string   `json:"authority"`
	Signature    string   `json:"signature"`
	EffectID     string   `json:"effect_id"`
	OutcomeLink  string   `json:"outcome_link"`
	EvidenceRefs []string `json:"evidence_refs,omitempty"`
}

type route struct {
	consumer  string
	authority string
	minSample int
}

var routes = map[string]route{
	"fak-guard-crash/1|TERMINAL_CRASH":              {"guard-crash-rsi", "authoritative", 1},
	"fak-guard-refusal/1|BLOCKED_BY_GUARD":          {"guard-audit", "authoritative", 2},
	"fak-benchmark-regression/1|QUALITY_REGRESSION": {"benchmark-bisect-profile", "authoritative", 1},
	"fak-contract-drift/1|PRODUCER_CONSUMER_STALE":  {"trigger-contract-audit", "authoritative", 1},
	"fak-intent-cluster/1|RECURRING_INTENT":         {"harness-garden", "heuristic", 3},
}

func main() {
	selfcheck := flag.Bool("selfcheck", false, "run deterministic built-in checks")
	nowText := flag.String("now", defaultNow, "evaluation time in RFC3339")
	flag.Parse()
	if *selfcheck {
		if err := runSelfcheck(); err != nil {
			fmt.Fprintln(os.Stderr, "selfcheck:", err)
			os.Exit(1)
		}
		fmt.Println("selfcheck: PASS (routing, signatures, recursion, duplicate, stale, unknown schema)")
		return
	}
	now, err := time.Parse(time.RFC3339, *nowText)
	if err != nil {
		fail("invalid -now: %v", err)
	}
	var r Receipt
	dec := json.NewDecoder(os.Stdin)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&r); err != nil {
		fail("decode bounded receipt: %v", err)
	}
	out := Evaluate(r, now, nil)
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(out); err != nil {
		fail("encode decision: %v", err)
	}
}

func Evaluate(in Receipt, now time.Time, activeEffects map[string]string) Decision {
	normalizedRefs := append([]string(nil), in.EvidenceRefs...)
	sort.Strings(normalizedRefs)
	key := in.Schema + "|" + in.Reason
	rt, known := routes[key]
	consumer, authority := "trigger-contract-audit", "unknown"
	if known {
		consumer, authority = rt.consumer, rt.authority
	}
	sig := stableHash(in.Schema, in.Reason, in.Producer, in.EffectKey, strings.Join(normalizedRefs, "\n"))
	effectID := "effect:" + stableHash(consumer, in.EffectKey)
	out := Decision{
		Schema: schemaDecision, Consumer: consumer, Authority: authority,
		Signature: sig, EffectID: effectID, OutcomeLink: "outcome:" + sig,
		EvidenceRefs: normalizedRefs,
	}
	if !known {
		out.Decision, out.Reason = "REROUTE", "UNKNOWN_PRODUCER_CONTRACT"
		return out
	}
	if in.Recursion {
		out.Decision, out.Reason = "SKIP", "RECURSION_SUPPRESSED"
		return out
	}
	produced, err := time.Parse(time.RFC3339, in.ProducedAt)
	if err != nil || produced.After(now.Add(time.Minute)) || now.Sub(produced) > 15*time.Minute {
		out.Decision, out.Reason = "DEFER", "INPUT_STALE"
		return out
	}
	if owner, exists := activeEffects[effectID]; exists {
		out.Decision, out.Reason = "MERGE", "DUPLICATE_EFFECT:"+owner
		return out
	}
	if in.Capacity < 1 {
		out.Decision, out.Reason = "DEFER", "NO_CAPACITY"
		return out
	}
	if rt.minSample > 1 && in.SampleCount < rt.minSample {
		out.Decision, out.Reason = "SKIP", "INSUFFICIENT_SAMPLE"
		return out
	}
	if in.ExpectedValue < 1 {
		out.Decision, out.Reason = "SKIP", "BELOW_VALUE_FLOOR"
		return out
	}
	out.Decision, out.Reason = "RUN", "MATCHED_RECEIPT_READY"
	return out
}

func stableHash(parts ...string) string {
	sum := sha256.Sum256([]byte(strings.Join(parts, "\x00")))
	return hex.EncodeToString(sum[:12])
}

func fail(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(2)
}

func runSelfcheck() error {
	now, _ := time.Parse(time.RFC3339, defaultNow)
	base := Receipt{Schema: "fak-guard-crash/1", Reason: "TERMINAL_CRASH", Producer: "guard", ProducedAt: "2026-08-31T11:59:00Z", EffectKey: "panic:ctxmmu", Capacity: 1, ExpectedValue: 5, SampleCount: 1, EvidenceRefs: []string{"receipt:abc"}}
	got := Evaluate(base, now, nil)
	if got.Decision != "RUN" || got.Consumer != "guard-crash-rsi" {
		return fmt.Errorf("exact routing: %+v", got)
	}
	reordered := base
	reordered.EvidenceRefs = []string{"z", "a"}
	reordered2 := base
	reordered2.EvidenceRefs = []string{"a", "z"}
	if Evaluate(reordered, now, nil).Signature != Evaluate(reordered2, now, nil).Signature {
		return fmt.Errorf("signature changed with evidence order")
	}
	recursive := base
	recursive.Recursion = true
	if d := Evaluate(recursive, now, nil); d.Decision != "SKIP" || d.Reason != "RECURSION_SUPPRESSED" {
		return fmt.Errorf("recursion: %+v", d)
	}
	active := map[string]string{got.EffectID: "run-17"}
	if d := Evaluate(base, now, active); d.Decision != "MERGE" {
		return fmt.Errorf("duplicate: %+v", d)
	}
	stale := base
	stale.ProducedAt = "2026-08-31T10:00:00Z"
	if d := Evaluate(stale, now, nil); d.Decision != "DEFER" || d.Reason != "INPUT_STALE" {
		return fmt.Errorf("stale: %+v", d)
	}
	unknown := base
	unknown.Schema = "mystery/9"
	if d := Evaluate(unknown, now, nil); d.Decision != "REROUTE" || d.Consumer != "trigger-contract-audit" {
		return fmt.Errorf("unknown: %+v", d)
	}
	cases := []Receipt{
		{Schema: "fak-guard-refusal/1", Reason: "BLOCKED_BY_GUARD", Producer: "guard", ProducedAt: base.ProducedAt, EffectKey: "guard:policy", Capacity: 1, ExpectedValue: 2, SampleCount: 2},
		{Schema: "fak-benchmark-regression/1", Reason: "QUALITY_REGRESSION", Producer: "bench", ProducedAt: base.ProducedAt, EffectKey: "bench:qwen38", Capacity: 1, ExpectedValue: 3},
		{Schema: "fak-contract-drift/1", Reason: "PRODUCER_CONSUMER_STALE", Producer: "trigger", ProducedAt: base.ProducedAt, EffectKey: "contract:x", Capacity: 1, ExpectedValue: 2},
		{Schema: "fak-intent-cluster/1", Reason: "RECURRING_INTENT", Producer: "trajectory", ProducedAt: base.ProducedAt, EffectKey: "intent:setup", Capacity: 1, ExpectedValue: 2, SampleCount: 3},
	}
	want := []string{"guard-audit", "benchmark-bisect-profile", "trigger-contract-audit", "harness-garden"}
	for i, c := range cases {
		if d := Evaluate(c, now, nil); d.Decision != "RUN" || d.Consumer != want[i] {
			return fmt.Errorf("route %d: %+v", i, d)
		}
	}
	return nil
}
