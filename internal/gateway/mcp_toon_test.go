package gateway

// Tests for the TOON wire (#3067) at the mcpToolResult seam. The load-bearing
// properties, in order: (1) default OFF — with FAK_TOON_WIRE unset the wire is a
// byte-identical no-op on every payload, including a perfect tabular candidate;
// (2) flag on, the #3066 gate fires only on a proven net win and the emitted text
// is exactly toon.Encode's output; (3) cache safety — a cache-resident span ships
// canonical JSON even when its shape would otherwise fire (CACHE_PREFIX_RESIDENT
// beats the shape gate), and a vDSO-served SyscallResponse is derived as resident;
// (4) volatility — a head carrying a per-request token ships JSON; (5) the golden
// corpus — a real captured `fak index leaf --json` payload ships byte-identical
// JSON on the skip path, and the flag-on decision over it is reported (observed
// token delta, feeds SCORE #3068).

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/toon"
)

// mcpResultText unwraps the single text content block mcpToolResult emits.
func mcpResultText(t *testing.T, m map[string]any) string {
	t.Helper()
	blocks, ok := m["content"].([]map[string]any)
	if !ok || len(blocks) != 1 {
		t.Fatalf("mcpToolResult content = %#v, want exactly one block", m["content"])
	}
	text, ok := blocks[0]["text"].(string)
	if !ok {
		t.Fatalf("mcpToolResult block carries no text: %#v", blocks[0])
	}
	return text
}

// fireShapedPayload is a json-native, uniform, flat tabular array big enough to pass
// every shape/size gate — the kind of payload the wire exists to win on. Built from
// map[string]any/[]any/float64 so the round-trip witness is provable.
func fireShapedPayload(rows int) any {
	out := make([]any, 0, rows)
	for i := 0; i < rows; i++ {
		out = append(out, map[string]any{
			"name":    fmt.Sprintf("leaf%02d", i),
			"tree":    fmt.Sprintf("internal/leaf%02d/**", i),
			"shipped": float64(i),
			"exists":  i%2 == 0,
		})
	}
	return out
}

// mustFire asserts the wire's own Decide inputs yield a FIRE on payload — guarding the
// tests below against silently passing because the payload stopped being fire-shaped.
func mustFire(t *testing.T, payload any) {
	t.Helper()
	b, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}
	d := toon.Decide(payload, toon.DecideInput{Volatile: agent.HeadValueIsVolatile(toonHeadRaw(b))})
	if !d.Encode {
		t.Fatalf("test payload no longer fire-shaped: %s", d)
	}
}

func TestMCPToolResultDefaultOffByteIdentical(t *testing.T) {
	t.Setenv("FAK_TOON_WIRE", "") // explicit: the no-flag arm

	payload := fireShapedPayload(12) // a perfect candidate — and still must ship JSON
	mustFire(t, payload)
	want, _ := json.Marshal(payload)
	if got := mcpResultText(t, mcpToolResult(payload)); got != string(want) {
		t.Fatalf("flag off: text diverged from canonical JSON\n got: %s\nwant: %s", got, want)
	}
}

func TestMCPToolResultFlagOnFiresOnTabularWin(t *testing.T) {
	t.Setenv("FAK_TOON_WIRE", "1")

	payload := fireShapedPayload(12)
	mustFire(t, payload)
	wantTOON, err := toon.Encode(payload, toon.Options{})
	if err != nil {
		t.Fatal(err)
	}
	got := mcpResultText(t, mcpToolResult(payload))
	if got != string(wantTOON) {
		t.Fatalf("flag on: text is not the TOON encoding\n got: %s\nwant: %s", got, wantTOON)
	}
	if wantJSON, _ := json.Marshal(payload); got == string(wantJSON) {
		t.Fatal("flag on: fire-shaped payload still shipped JSON")
	}
}

func TestMCPToolResultCacheResidentShipsJSON(t *testing.T) {
	t.Setenv("FAK_TOON_WIRE", "1")

	// A perfect tabular candidate marked cache-resident: CACHE_PREFIX_RESIDENT must
	// win over the shape gate and ship byte-identical JSON.
	payload := fireShapedPayload(12)
	mustFire(t, payload)
	want, _ := json.Marshal(payload)
	if got := mcpResultText(t, mcpToolResultSpan(payload, true)); got != string(want) {
		t.Fatalf("cache-resident span was re-encoded\n got: %s\nwant: %s", got, want)
	}

	// End-to-end signal derivation: a vDSO-served SyscallResponse is resident.
	resp := SyscallResponse{Verdict: WireVerdict{Kind: "ALLOW", By: "vdso"}}
	if !toonCacheResident(resp) || !toonCacheResident(&resp) {
		t.Fatal("vDSO-served SyscallResponse not derived as cache-resident")
	}
	if toonCacheResident(SyscallResponse{Verdict: WireVerdict{Kind: "ALLOW", By: "policy"}}) {
		t.Fatal("non-vDSO SyscallResponse wrongly derived as cache-resident")
	}
	wantResp, _ := json.Marshal(resp)
	if got := mcpResultText(t, mcpToolResult(resp)); got != string(wantResp) {
		t.Fatalf("vDSO-served response diverged from canonical JSON\n got: %s\nwant: %s", got, wantResp)
	}
}

func TestMCPToolResultVolatileHeadShipsJSON(t *testing.T) {
	t.Setenv("FAK_TOON_WIRE", "1")

	// Uniform, flat, big — but every row carries a per-request UUID, so the head
	// element is volatile (the same evidence anthropic_cachebp.go refuses to anchor
	// a breakpoint on) and the span ships JSON.
	rows := make([]any, 0, 12)
	for i := 0; i < 12; i++ {
		rows = append(rows, map[string]any{
			"id":   fmt.Sprintf("3f2a8c1e-9b4d-4e6a-8c2f-1d5e7a9b3c%02d", i),
			"name": fmt.Sprintf("leaf%02d", i),
			"n":    float64(i),
		})
	}
	want, _ := json.Marshal(rows)
	if got := mcpResultText(t, mcpToolResult(any(rows))); got != string(want) {
		t.Fatalf("volatile-head span was re-encoded\n got: %s\nwant: %s", got, want)
	}
}

// TestMCPToolResultGoldenIndexLeavesCorpus runs the wire over a REAL captured
// tool-result corpus (`fak index leaf --json`, frozen in testdata) — the golden
// witness that the skip path is unchanged, plus the observed flag-on token delta
// the SCORE loop (#3068) feeds on.
func TestMCPToolResultGoldenIndexLeavesCorpus(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "toon_corpus_index_leaves.json"))
	if err != nil {
		t.Fatal(err)
	}
	var payload any
	if err := json.Unmarshal(raw, &payload); err != nil {
		t.Fatal(err)
	}
	want, _ := json.Marshal(payload) // canonical compact form of the same bytes

	t.Setenv("FAK_TOON_WIRE", "")
	if got := mcpResultText(t, mcpToolResult(payload)); got != string(want) {
		t.Fatal("golden corpus: flag-off text diverged from canonical JSON")
	}

	t.Setenv("FAK_TOON_WIRE", "1")
	d := toon.Decide(payload, toon.DecideInput{Volatile: agent.HeadValueIsVolatile(toonHeadRaw(want))})
	t.Logf("observed on real corpus (flag on): %s (net %+d tokens)", d, d.TOONTokens-d.JSONTokens)
	got := mcpResultText(t, mcpToolResult(payload))
	if d.Encode {
		if got == string(want) {
			t.Fatal("golden corpus: gate fired but the wire shipped JSON")
		}
		if d.TOONTokens >= d.JSONTokens {
			t.Fatalf("never-fire-at-a-loss violated: %s", d)
		}
	} else if got != string(want) {
		t.Fatalf("golden corpus: gate skipped (%s) but the wire re-encoded", d.Reason)
	}
}
