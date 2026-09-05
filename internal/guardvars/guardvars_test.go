package guardvars

import (
	"encoding/json"
	"reflect"
	"slices"
	"sort"
	"testing"
)

// This package exists so the /debug/vars PRODUCER (internal/gateway) and the `fak info`
// CONSUMER (cmd/fak) share ONE definition of each wire block instead of two hand-copied
// structs that silently drift. Aliasing both sides to one struct makes a MISSING FIELD a
// compile-time impossibility — but it does nothing about the JSON TAG SPELLING: renaming
// `json:"tokens_left"` to `json:"tokens_lef"` still compiles on both sides while the
// consumer decodes a zero forever. These tests are the guard for that failure mode.
//
// The key sets below are written out as literals ON PURPOSE. Deriving them by reflecting
// over the struct tags would produce a test that reads the tags to check the tags: it
// would pass no matter what they say.

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

// populatedSessionVars fills EVERY field with a distinct non-zero value so that (a) no
// `omitempty` key can hide from the wire-key assertion and (b) a tag accidentally swapped
// between two same-typed fields changes an observable value rather than staying invisible.
//
// It is deliberately NOT a realistic row: the type doc notes a real row carries
// InflightSeconds or IdleSeconds, never both. That invariant belongs to the producer
// (internal/gateway); this leaf is pure data and cannot enforce it. Here we want maximum
// wire coverage, so both are set.
func populatedSessionVars() SessionVars {
	return SessionVars{
		TraceID:           "trace-alpha",
		Run:               "run-beta",
		ParentTrace:       "trace-parent",
		Generation:        3,
		Priority:          5,
		TurnsLeft:         7,
		TokensLeft:        11,
		ContextTokensLeft: 13,
		ElapsedSeconds:    17,
		Assumptions:       19,
		LastTool:          "Bash",
		SpawnCount:        23,
		InflightSeconds:   29,
		IdleSeconds:       31,
	}
}

// sessionVarsWire is the hand-written wire form of populatedSessionVars, with the keys in a
// scrambled order so nothing here depends on field declaration order. Decoding this literal
// and comparing against the fixture pins the tag spelling from the CONSUMER direction, which
// a marshal→unmarshal round-trip cannot do: swapping the tags of two same-typed fields
// round-trips perfectly and is only caught by a fixed payload like this one.
const sessionVarsWire = `{
	"idle_seconds": 31,
	"tokens_left": 11,
	"parent_trace": "trace-parent",
	"last_tool": "Bash",
	"turns_left": 7,
	"generation": 3,
	"run": "run-beta",
	"context_tokens_left": 13,
	"spawn_count": 23,
	"priority": 5,
	"trace_id": "trace-alpha",
	"inflight_seconds": 29,
	"assumptions": 19,
	"elapsed_seconds": 17
}`

// populatedCacheAttributionVars fills every field with a distinct non-zero value. The write
// premium is negative on purpose: the type doc says it stays negative until reads repay
// writes, and a float tag that silently lost its sign would be a real consumer bug.
func populatedCacheAttributionVars() CacheAttributionVars {
	return CacheAttributionVars{
		ProviderTokenEquiv: 1.5,
		FakTokenEquiv:      2.25,
		TotalTokenEquiv:    3.75,

		ProviderPromptCacheReadTokenEquiv:         8.5,
		ProviderPromptCacheWritePremiumTokenEquiv: -6.25,
		CacheCreationTokensHeadOnly:               17,
		CacheCreationTokensMessagePrefix:          19,
		FakCompactionShedTokens:                   101,
		FakCompactionCacheReadTokens:              103,
		FakKVPrefixReusedTokens:                   107,
		FakVDSOAvoidedCalls:                       109,
		FakInlineServedCalls:                      111,
		FakResponseMemoCalls:                      112,

		FakDeferColdTurns:      113,
		FakDeferColdCount:      127,
		FakDeferColdToolNames:  []string{"Bash", "Grep"},
		FakDeferStandDownTurns: 131,
		FakDeferFinding:        FindingDeferEnabledButInert,
	}
}

const cacheAttributionVarsWire = `{
	"fak_defer_cold_count": 127,
	"total_token_equiv": 3.75,
	"fak_vdso_avoided_calls": 109,
	"fak_inline_served_calls": 111,
	"fak_response_memo_calls": 112,
	"provider_prompt_cache_write_premium_token_equiv": -6.25,
	"cache_creation_tokens_head_only": 17,
	"cache_creation_tokens_message_prefix": 19,
	"fak_compaction_shed_tokens": 101,
	"fak_defer_finding": "DEFER_ENABLED_BUT_INERT",
	"provider_token_equiv": 1.5,
	"fak_kv_prefix_reused_tokens": 107,
	"fak_defer_cold_turns": 113,
	"provider_prompt_cache_read_token_equiv": 8.5,
	"fak_token_equiv": 2.25,
	"fak_compaction_cache_read_tokens": 103,
	"fak_defer_stand_down_turns": 131,
	"fak_defer_cold_tool_names": ["Bash", "Grep"]
}`

// populatedManagedCacheVars sets Active true and Inert false so the two adjacent bools are
// DISTINGUISHABLE: with both true, a tag swap between them would be invisible.
func populatedManagedCacheVars() ManagedCacheVars {
	return ManagedCacheVars{
		Active:   true,
		Inert:    false,
		Upgraded: 41,
		Reasons:  map[string]uint64{"head_too_small": 2, "no_cache_control": 3},
		Wire:     WireOpenAIResponses,
		Finding:  FindingUpgradeNeverFired,
	}
}

const managedCacheVarsWire = `{
	"wire": "openai-responses",
	"upgraded": 41,
	"inert": false,
	"finding": "UPGRADE_NEVER_FIRED",
	"active": true,
	"reasons": {"head_too_small": 2, "no_cache_control": 3}
}`

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// wireKeys marshals v and returns the sorted top-level JSON object keys it emitted.
func wireKeys(t *testing.T, v any) []string {
	t.Helper()
	raw, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("json.Marshal(%T): %v", v, err)
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil {
		t.Fatalf("re-decoding marshaled %T (%s): %v", v, raw, err)
	}
	keys := make([]string, 0, len(obj))
	for k := range obj {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// assertKeys compares an emitted key set against the explicitly named expectation and
// reports each missing and each unexpected key by name, because "a tag was renamed" and
// "a field was dropped" have to be told apart from the failure text alone.
func assertKeys(t *testing.T, what string, got, want []string) {
	t.Helper()
	sorted := append([]string(nil), want...)
	sort.Strings(sorted)
	if reflect.DeepEqual(got, sorted) {
		return
	}
	inGot := make(map[string]bool, len(got))
	for _, k := range got {
		inGot[k] = true
	}
	inWant := make(map[string]bool, len(sorted))
	for _, k := range sorted {
		inWant[k] = true
	}
	for _, k := range sorted {
		if !inGot[k] {
			t.Errorf("%s: wire key %q is MISSING (tag renamed, field dropped, or made omitempty)", what, k)
		}
	}
	for _, k := range got {
		if !inWant[k] {
			t.Errorf("%s: wire key %q is UNEXPECTED (tag renamed or field added without updating this pin)", what, k)
		}
	}
	t.Errorf("%s: emitted keys\n got  %q\n want %q", what, got, sorted)
}

// decodeWire unmarshals a literal wire payload into a fresh value of type T.
func decodeWire[T any](t *testing.T, payload string) T {
	t.Helper()
	var out T
	if err := json.Unmarshal([]byte(payload), &out); err != nil {
		t.Fatalf("json.Unmarshal into %T: %v", out, err)
	}
	return out
}

// ---------------------------------------------------------------------------
// 1. Wire-contract pinning: the exact JSON key set of each block
// ---------------------------------------------------------------------------

func TestSessionVarsWireKeys(t *testing.T) {
	assertKeys(t, "SessionVars", wireKeys(t, populatedSessionVars()), []string{
		"trace_id",
		"run",
		"parent_trace",
		"generation",
		"priority",
		"turns_left",
		"tokens_left",
		"context_tokens_left",
		"elapsed_seconds",
		"assumptions",
		"last_tool",
		"spawn_count",
		"inflight_seconds",
		"idle_seconds",
	})
}

func TestCacheAttributionVarsWireKeys(t *testing.T) {
	assertKeys(t, "CacheAttributionVars", wireKeys(t, populatedCacheAttributionVars()), []string{
		"provider_token_equiv",
		"fak_token_equiv",
		"total_token_equiv",
		"provider_prompt_cache_read_token_equiv",
		"provider_prompt_cache_write_premium_token_equiv",
		"cache_creation_tokens_head_only",
		"cache_creation_tokens_message_prefix",
		"fak_compaction_shed_tokens",
		"fak_compaction_cache_read_tokens",
		"fak_kv_prefix_reused_tokens",
		"fak_vdso_avoided_calls",
		"fak_inline_served_calls",
		"fak_response_memo_calls",
		"fak_defer_cold_turns",
		"fak_defer_cold_count",
		"fak_defer_cold_tool_names",
		"fak_defer_stand_down_turns",
		"fak_defer_finding",
	})
}

func TestManagedCacheVarsWireKeys(t *testing.T) {
	assertKeys(t, "ManagedCacheVars", wireKeys(t, populatedManagedCacheVars()), []string{
		"active",
		"inert",
		"upgraded",
		"reasons",
		"wire",
		"finding",
	})
}

// TestWireKeysCarryTheRightValues pins key→VALUE, not just the key set. A key-set assertion
// alone cannot see two tags swapped between same-typed fields; comparing the whole emitted
// object against a hand-written expectation can.
func TestWireKeysCarryTheRightValues(t *testing.T) {
	for _, tc := range []struct {
		name  string
		value any
		want  map[string]any
	}{
		{
			name:  "SessionVars",
			value: populatedSessionVars(),
			want: map[string]any{
				"trace_id":            "trace-alpha",
				"run":                 "run-beta",
				"parent_trace":        "trace-parent",
				"generation":          float64(3),
				"priority":            float64(5),
				"turns_left":          float64(7),
				"tokens_left":         float64(11),
				"context_tokens_left": float64(13),
				"elapsed_seconds":     float64(17),
				"assumptions":         float64(19),
				"last_tool":           "Bash",
				"spawn_count":         float64(23),
				"inflight_seconds":    float64(29),
				"idle_seconds":        float64(31),
			},
		},
		{
			name:  "CacheAttributionVars",
			value: populatedCacheAttributionVars(),
			want: map[string]any{
				"provider_token_equiv":                            1.5,
				"fak_token_equiv":                                 2.25,
				"total_token_equiv":                               3.75,
				"provider_prompt_cache_read_token_equiv":          8.5,
				"provider_prompt_cache_write_premium_token_equiv": -6.25,
				"cache_creation_tokens_head_only":                 float64(17),
				"cache_creation_tokens_message_prefix":            float64(19),
				"fak_compaction_shed_tokens":                      float64(101),
				"fak_compaction_cache_read_tokens":                float64(103),
				"fak_kv_prefix_reused_tokens":                     float64(107),
				"fak_vdso_avoided_calls":                          float64(109),
				"fak_inline_served_calls":                         float64(111),
				"fak_response_memo_calls":                         float64(112),
				"fak_defer_cold_turns":                            float64(113),
				"fak_defer_cold_count":                            float64(127),
				"fak_defer_cold_tool_names":                       []any{"Bash", "Grep"},
				"fak_defer_stand_down_turns":                      float64(131),
				"fak_defer_finding":                               "DEFER_ENABLED_BUT_INERT",
			},
		},
		{
			name:  "ManagedCacheVars",
			value: populatedManagedCacheVars(),
			want: map[string]any{
				"active":   true,
				"inert":    false,
				"upgraded": float64(41),
				"reasons":  map[string]any{"head_too_small": float64(2), "no_cache_control": float64(3)},
				"wire":     "openai-responses",
				"finding":  "UPGRADE_NEVER_FIRED",
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			raw, err := json.Marshal(tc.value)
			if err != nil {
				t.Fatalf("json.Marshal: %v", err)
			}
			var got map[string]any
			if err := json.Unmarshal(raw, &got); err != nil {
				t.Fatalf("re-decoding %s: %v", raw, err)
			}
			for k, wantV := range tc.want {
				gotV, ok := got[k]
				if !ok {
					t.Errorf("key %q absent from the wire", k)
					continue
				}
				if !reflect.DeepEqual(gotV, wantV) {
					t.Errorf("key %q = %#v, want %#v (tags swapped between two fields?)", k, gotV, wantV)
				}
			}
			for k := range got {
				if _, ok := tc.want[k]; !ok {
					t.Errorf("unexpected wire key %q", k)
				}
			}
		})
	}
}

// TestFieldCountsMatchThePinnedKeySets is the tripwire the key-set tests cannot be: a field
// added with `json:"-"` emits no key at all, so every assertion above would still pass while
// a new field entered the shared shape unreviewed. This counts fields; it never reads a tag,
// so it cannot degenerate into checking the tags against themselves.
func TestFieldCountsMatchThePinnedKeySets(t *testing.T) {
	for _, tc := range []struct {
		name string
		typ  reflect.Type
		want int
	}{
		{"SessionVars", reflect.TypeOf(SessionVars{}), 14},
		{"CacheAttributionVars", reflect.TypeOf(CacheAttributionVars{}), 18},
		{"ManagedCacheVars", reflect.TypeOf(ManagedCacheVars{}), 6},
	} {
		if got := tc.typ.NumField(); got != tc.want {
			t.Errorf("%s has %d fields, want %d — a field was added or removed; update the wire-key pin above to match (a `json:\"-\"` field would otherwise slip past it)", tc.name, got, tc.want)
		}
	}
}

// TestWireTokenConstantsAreStable pins the three cross-process string tokens. They are not
// private to this package: the gateway writes them onto /debug/vars, the `fak info` pane and
// the guard exit banner match on them, and internal/sessionaudit re-states the wire string by
// value to stay stdlib-only. Changing a constant's VALUE (as opposed to its Go identifier)
// silently desynchronises those surfaces, so the literal is part of the contract.
func TestWireTokenConstantsAreStable(t *testing.T) {
	for _, tc := range []struct{ name, got, want string }{
		{"WireOpenAIResponses", WireOpenAIResponses, "openai-responses"},
		{"FindingDeferEnabledButInert", FindingDeferEnabledButInert, "DEFER_ENABLED_BUT_INERT"},
		{"FindingUpgradeNeverFired", FindingUpgradeNeverFired, "UPGRADE_NEVER_FIRED"},
	} {
		if tc.got != tc.want {
			t.Errorf("%s = %q, want %q", tc.name, tc.got, tc.want)
		}
	}
}

// ---------------------------------------------------------------------------
// 2. omitempty: marshaling only, never unmarshaling
// ---------------------------------------------------------------------------

// TestZeroValueEmitsExactlyTheMandatoryKeys pins the producer half of the doc's omitempty
// claim. The comparison is on exact BYTES, not a key set: the type doc says omitempty keeps
// the block byte-stable, and for a zero value that byte sequence is short enough to name.
// Reordering fields in the struct is therefore an intentional, reviewable change here.
func TestZeroValueEmitsExactlyTheMandatoryKeys(t *testing.T) {
	for _, tc := range []struct {
		name  string
		value any
		want  string
	}{
		{
			name:  "SessionVars",
			value: SessionVars{},
			want:  `{"trace_id":"","run":"","turns_left":0,"tokens_left":0}`,
		},
		{
			name:  "CacheAttributionVars",
			value: CacheAttributionVars{},
			want: `{"provider_token_equiv":0,"fak_token_equiv":0,"total_token_equiv":0,` +
				`"provider_prompt_cache_read_token_equiv":0,` +
				`"provider_prompt_cache_write_premium_token_equiv":0,` +
				`"fak_compaction_shed_tokens":0,"fak_kv_prefix_reused_tokens":0,` +
				`"fak_vdso_avoided_calls":0,"fak_response_memo_calls":0,"fak_inline_served_calls":0}`,
		},
		{
			name:  "ManagedCacheVars",
			value: ManagedCacheVars{},
			want:  `{"active":false,"inert":false,"upgraded":0}`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			raw, err := json.Marshal(tc.value)
			if err != nil {
				t.Fatalf("json.Marshal: %v", err)
			}
			if string(raw) != tc.want {
				t.Errorf("zero-value wire mismatch\n got  %s\n want %s", raw, tc.want)
			}
		})
	}
}

// TestEmptyReferenceFieldsAreOmitted covers the reference-typed omitempty fields, where
// "empty" is not the same as "nil": an allocated-but-empty map or slice is still omitted.
// A future reader tempted to drop the omitempty on Reasons would emit `"reasons":{}` on every
// Anthropic-wire block, which is exactly the byte instability the type doc rules out.
func TestEmptyReferenceFieldsAreOmitted(t *testing.T) {
	mc := ManagedCacheVars{Active: true, Reasons: map[string]uint64{}}
	if got := string(mustMarshal(t, mc)); got != `{"active":true,"inert":false,"upgraded":0}` {
		t.Errorf("empty Reasons map leaked onto the wire: %s", got)
	}
	mc.Reasons = map[string]uint64{"head_too_small": 1}
	if got := string(mustMarshal(t, mc)); got != `{"active":true,"inert":false,"upgraded":0,"reasons":{"head_too_small":1}}` {
		t.Errorf("non-empty Reasons wire mismatch: %s", got)
	}

	ca := CacheAttributionVars{FakDeferColdToolNames: []string{}}
	if keys := wireKeys(t, ca); slices.Contains(keys, "fak_defer_cold_tool_names") {
		t.Errorf("empty FakDeferColdToolNames slice leaked onto the wire: %q", keys)
	}
	ca.FakDeferColdToolNames = []string{"Bash"}
	if keys := wireKeys(t, ca); !slices.Contains(keys, "fak_defer_cold_tool_names") {
		t.Errorf("non-empty FakDeferColdToolNames was dropped from the wire: %q", keys)
	}
}

// TestOmitemptyDoesNotAffectDecoding pins the consumer half of the doc's claim: omitempty is
// a marshal option only, so a payload that omits every optional key still decodes, and one
// that carries them populates them. This is the exact drift the package was created to stop —
// a renamed tag decodes as a permanent zero here rather than as an error.
func TestOmitemptyDoesNotAffectDecoding(t *testing.T) {
	t.Run("SessionVars/mandatory-only", func(t *testing.T) {
		got := decodeWire[SessionVars](t, `{"trace_id":"t-1","run":"r-1","turns_left":4,"tokens_left":9}`)
		want := SessionVars{TraceID: "t-1", Run: "r-1", TurnsLeft: 4, TokensLeft: 9}
		if got != want {
			t.Errorf("decoded %+v, want %+v", got, want)
		}
	})

	t.Run("SessionVars/full", func(t *testing.T) {
		got := decodeWire[SessionVars](t, sessionVarsWire)
		if want := populatedSessionVars(); got != want {
			t.Errorf("decoded %+v, want %+v (a renamed tag decodes as a silent zero)", got, want)
		}
	})

	t.Run("CacheAttributionVars/mandatory-only", func(t *testing.T) {
		got := decodeWire[CacheAttributionVars](t, `{
			"provider_token_equiv": 1,
			"fak_token_equiv": 2,
			"total_token_equiv": 3,
			"provider_prompt_cache_read_token_equiv": 4,
			"provider_prompt_cache_write_premium_token_equiv": -5,
			"fak_compaction_shed_tokens": 6,
			"fak_kv_prefix_reused_tokens": 7,
			"fak_vdso_avoided_calls": 8
		}`)
		want := CacheAttributionVars{
			ProviderTokenEquiv:                        1,
			FakTokenEquiv:                             2,
			TotalTokenEquiv:                           3,
			ProviderPromptCacheReadTokenEquiv:         4,
			ProviderPromptCacheWritePremiumTokenEquiv: -5,
			FakCompactionShedTokens:                   6,
			FakKVPrefixReusedTokens:                   7,
			FakVDSOAvoidedCalls:                       8,
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("decoded %+v, want %+v", got, want)
		}
	})

	t.Run("CacheAttributionVars/full", func(t *testing.T) {
		got := decodeWire[CacheAttributionVars](t, cacheAttributionVarsWire)
		if want := populatedCacheAttributionVars(); !reflect.DeepEqual(got, want) {
			t.Errorf("decoded %+v, want %+v", got, want)
		}
	})

	t.Run("ManagedCacheVars/mandatory-only", func(t *testing.T) {
		got := decodeWire[ManagedCacheVars](t, `{"active":true,"inert":true,"upgraded":6}`)
		want := ManagedCacheVars{Active: true, Inert: true, Upgraded: 6}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("decoded %+v, want %+v", got, want)
		}
	})

	t.Run("ManagedCacheVars/full", func(t *testing.T) {
		got := decodeWire[ManagedCacheVars](t, managedCacheVarsWire)
		if want := populatedManagedCacheVars(); !reflect.DeepEqual(got, want) {
			t.Errorf("decoded %+v, want %+v", got, want)
		}
	})
}

// ---------------------------------------------------------------------------
// 3. WireHasNo1hTTLLever — the one behavioural function
// ---------------------------------------------------------------------------

// TestWireHasNo1hTTLLever covers the three states the type doc names — the OpenAI Responses
// wire (no 1h-TTL lever), the empty historical default (Anthropic reading preserved), and any
// other wire — plus the near-misses that would pass if the exact-match test were ever
// loosened into a prefix or substring or case-insensitive comparison.
func TestWireHasNo1hTTLLever(t *testing.T) {
	for _, tc := range []struct {
		name string
		wire string
		want bool
	}{
		{"openai-responses wire has no 1h lever", WireOpenAIResponses, true},
		{"empty wire preserves the Anthropic reading", "", false},
		{"anthropic wire has the lever", "anthropic", false},
		{"anthropic-messages wire has the lever", "anthropic-messages", false},
		{"plain openai wire is not the responses wire", "openai", false},
		{"case differs", "OpenAI-Responses", false},
		{"longer wire sharing the prefix", "openai-responses-preview", false},
		{"leading space is not trimmed", " openai-responses", false},
		{"substring inside another wire", "azure/openai-responses/v1", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := ManagedCacheVars{Wire: tc.wire}
			if got := m.WireHasNo1hTTLLever(); got != tc.want {
				t.Errorf("ManagedCacheVars{Wire: %q}.WireHasNo1hTTLLever() = %v, want %v", tc.wire, got, tc.want)
			}
		})
	}
}

// TestWireHasNo1hTTLLeverIgnoresTheOtherFields pins that the verdict is a function of Wire
// alone: the posture fields must not be able to talk it into or out of an answer, since the
// producer calls it while it is still COMPUTING Inert.
func TestWireHasNo1hTTLLeverIgnoresTheOtherFields(t *testing.T) {
	loud := ManagedCacheVars{
		Active:   true,
		Inert:    true,
		Upgraded: 99,
		Reasons:  map[string]uint64{"head_too_small": 4},
		Finding:  FindingUpgradeNeverFired,
	}
	quiet := ManagedCacheVars{}
	for _, wire := range []string{WireOpenAIResponses, "", "anthropic"} {
		loud.Wire, quiet.Wire = wire, wire
		if loud.WireHasNo1hTTLLever() != quiet.WireHasNo1hTTLLever() {
			t.Errorf("wire %q: verdict changed with the non-wire fields (%v vs %v)",
				wire, loud.WireHasNo1hTTLLever(), quiet.WireHasNo1hTTLLever())
		}
	}
}

// ---------------------------------------------------------------------------
// 4. Round trip — the producer→consumer path this package models
// ---------------------------------------------------------------------------

// TestRoundTripPreservesEveryField walks the whole modelled path: producer marshals, consumer
// unmarshals, nothing changes. Note the limit — a round trip CANNOT catch two tags swapped
// between same-typed fields, because the swap cancels itself out. That case is covered by
// TestOmitemptyDoesNotAffectDecoding's fixed-payload subtests and by
// TestWireKeysCarryTheRightValues; this test is the end-to-end check on top of them.
func TestRoundTripPreservesEveryField(t *testing.T) {
	t.Run("SessionVars", func(t *testing.T) {
		want := populatedSessionVars()
		got := decodeWire[SessionVars](t, string(mustMarshal(t, want)))
		if got != want {
			t.Errorf("round trip lost data\n got  %+v\n want %+v", got, want)
		}
	})

	t.Run("CacheAttributionVars", func(t *testing.T) {
		want := populatedCacheAttributionVars()
		got := decodeWire[CacheAttributionVars](t, string(mustMarshal(t, want)))
		if !reflect.DeepEqual(got, want) {
			t.Errorf("round trip lost data\n got  %+v\n want %+v", got, want)
		}
	})

	t.Run("ManagedCacheVars", func(t *testing.T) {
		want := populatedManagedCacheVars()
		got := decodeWire[ManagedCacheVars](t, string(mustMarshal(t, want)))
		if !reflect.DeepEqual(got, want) {
			t.Errorf("round trip lost data\n got  %+v\n want %+v", got, want)
		}
		if got.WireHasNo1hTTLLever() != want.WireHasNo1hTTLLever() {
			t.Errorf("round trip changed the wire verdict: %v -> %v",
				want.WireHasNo1hTTLLever(), got.WireHasNo1hTTLLever())
		}
	})

	t.Run("zero values round trip too", func(t *testing.T) {
		var sv SessionVars
		if got := decodeWire[SessionVars](t, string(mustMarshal(t, sv))); got != sv {
			t.Errorf("SessionVars zero round trip: %+v", got)
		}
		var mc ManagedCacheVars
		if got := decodeWire[ManagedCacheVars](t, string(mustMarshal(t, mc))); !reflect.DeepEqual(got, mc) {
			t.Errorf("ManagedCacheVars zero round trip: %+v", got)
		}
		var ca CacheAttributionVars
		if got := decodeWire[CacheAttributionVars](t, string(mustMarshal(t, ca))); !reflect.DeepEqual(got, ca) {
			t.Errorf("CacheAttributionVars zero round trip: %+v", got)
		}
	})
}

func mustMarshal(t *testing.T, v any) []byte {
	t.Helper()
	raw, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("json.Marshal(%T): %v", v, err)
	}
	return raw
}

func TestObservationEnvelopeGoldenAndStateLaws(t *testing.T) {
	observedZero := ObservationEnvelope{Schema: ObservationSchemaV1, Source: "gateway", Revision: "r1", Provenance: "measured", Availability: AvailabilityObserved, Data: json.RawMessage(`{"hits":0}`)}
	if err := observedZero.Validate(); err != nil {
		t.Fatalf("measured zero: %v", err)
	}
	got, err := json.Marshal(observedZero)
	if err != nil {
		t.Fatal(err)
	}
	want := `{"schema":"fak-observation/1","source":"gateway","revision":"r1","provenance":"measured","availability":"OBSERVED","data":{"hits":0}}`
	if string(got) != want {
		t.Fatalf("golden = %s, want %s", got, want)
	}

	valid := []ObservationEnvelope{
		{Schema: ObservationSchemaV1, Source: "gateway", Revision: "r1", Provenance: "measured", Availability: AvailabilityEmpty},
		{Schema: ObservationSchemaV1, Source: "gateway", ObservedAt: "2026-08-21T00:00:00Z", Provenance: "probe", Availability: AvailabilityUnavailable, Reason: "probe failed"},
		{Schema: ObservationSchemaV1, Source: "gateway", Revision: "r1", Provenance: "cache", Availability: AvailabilityStale, Reason: "ttl elapsed"},
		{Schema: ObservationSchemaV1, Source: "gateway", Revision: "r1", Provenance: "platform", Availability: AvailabilityNotApplicable, Reason: "unsupported platform"},
	}
	for _, envelope := range valid {
		if err := envelope.Validate(); err != nil {
			t.Errorf("%s: %v", envelope.Availability, err)
		}
	}

	invalid := []ObservationEnvelope{
		{Schema: "fak-observation/2", Source: "gateway", Revision: "r1", Provenance: "measured", Availability: AvailabilityEmpty},
		{Schema: ObservationSchemaV1, Source: "gateway", Revision: "r1", Provenance: "measured", Availability: Availability("FUTURE")},
		{Schema: ObservationSchemaV1, Source: "gateway", Revision: "r1", Provenance: "probe", Availability: AvailabilityUnavailable, Data: json.RawMessage(`0`)},
		{Schema: ObservationSchemaV1, Source: "gateway", Revision: "r1", Provenance: "cache", Availability: AvailabilityStale},
	}
	for i, envelope := range invalid {
		if err := envelope.Validate(); err == nil {
			t.Errorf("invalid case %d accepted: %+v", i, envelope)
		}
	}
}

// ---------------------------------------------------------------------------
// 5. Benchmarks
// ---------------------------------------------------------------------------

var (
	benchSinkBytes []byte
	benchSinkErr   error
	benchSinkBool  bool
)

func populatedShrinkLeverVars() ShrinkLeverVars {
	return ShrinkLeverVars{
		WireRunsLevers:   true,
		Wire:             "anthropic",
		LiveOnWire:       []string{ShrinkLeverCompactHistoryBudget, ShrinkLeverElideStaleReads},
		InertOnWire:      []string{ShrinkLeverDeferColdTools},
		DualLocalRouting: false,
		Finding:          FindingShrinkLeverInertOnWire,
	}
}

const shrinkLeverVarsWire = `{
	"wire_runs_levers": true,
	"wire": "anthropic",
	"live_on_wire": ["compact_history_budget", "elide_stale_reads"],
	"inert_on_wire": ["defer_cold_tools"],
	"dual_local_routing": false,
	"finding": "SHRINK_LEVER_INERT_ON_WIRE"
}`

func BenchmarkSessionVars_Marshal(b *testing.B) {
	sv := populatedSessionVars()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err := json.Marshal(sv)
		if err != nil {
			b.Fatal(err)
		}
		benchSinkBytes = out
	}
}

func BenchmarkSessionVars_Unmarshal(b *testing.B) {
	raw := []byte(sessionVarsWire)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var sv SessionVars
		if err := json.Unmarshal(raw, &sv); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkCacheAttributionVars_Marshal(b *testing.B) {
	ca := populatedCacheAttributionVars()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err := json.Marshal(ca)
		if err != nil {
			b.Fatal(err)
		}
		benchSinkBytes = out
	}
}

func BenchmarkCacheAttributionVars_Unmarshal(b *testing.B) {
	raw := []byte(cacheAttributionVarsWire)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var ca CacheAttributionVars
		if err := json.Unmarshal(raw, &ca); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkManagedCacheVars_Marshal(b *testing.B) {
	mc := populatedManagedCacheVars()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err := json.Marshal(mc)
		if err != nil {
			b.Fatal(err)
		}
		benchSinkBytes = out
	}
}

func BenchmarkManagedCacheVars_Unmarshal(b *testing.B) {
	raw := []byte(managedCacheVarsWire)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var mc ManagedCacheVars
		if err := json.Unmarshal(raw, &mc); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkManagedCacheVars_WireHasNo1hTTLLever(b *testing.B) {
	mc := ManagedCacheVars{Wire: WireOpenAIResponses}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkBool = mc.WireHasNo1hTTLLever()
	}
}

func BenchmarkShrinkLeverVars_Marshal(b *testing.B) {
	sl := populatedShrinkLeverVars()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err := json.Marshal(sl)
		if err != nil {
			b.Fatal(err)
		}
		benchSinkBytes = out
	}
}

func BenchmarkShrinkLeverVars_Unmarshal(b *testing.B) {
	raw := []byte(shrinkLeverVarsWire)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var sl ShrinkLeverVars
		if err := json.Unmarshal(raw, &sl); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkObservationEnvelope_Validate(b *testing.B) {
	envelope := ObservationEnvelope{
		Schema:       ObservationSchemaV1,
		Source:       "gateway",
		Revision:     "r1",
		Provenance:   "measured",
		Availability: AvailabilityObserved,
		Data:         json.RawMessage(`{"hits":0}`),
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkErr = envelope.Validate()
	}
}

func BenchmarkObservationEnvelope_Marshal(b *testing.B) {
	envelope := ObservationEnvelope{
		Schema:       ObservationSchemaV1,
		Source:       "gateway",
		Revision:     "r1",
		Provenance:   "measured",
		Availability: AvailabilityObserved,
		Data:         json.RawMessage(`{"hits":0}`),
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err := json.Marshal(envelope)
		if err != nil {
			b.Fatal(err)
		}
		benchSinkBytes = out
	}
}

func BenchmarkObservationEnvelope_Unmarshal(b *testing.B) {
	raw := []byte(`{"schema":"fak-observation/1","source":"gateway","revision":"r1","provenance":"measured","availability":"OBSERVED","data":{"hits":0}}`)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var env ObservationEnvelope
		if err := json.Unmarshal(raw, &env); err != nil {
			b.Fatal(err)
		}
	}
}
