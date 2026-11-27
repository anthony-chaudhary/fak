package gateway

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func TestModelCacheControl_Parse(t *testing.T) {
	t.Run("fast path no markers", func(t *testing.T) {
		in := `{"file_path":"a.go","limit":10}`
		forceFresh, maxAgeMS, hasMaxAge, reuseOnly, stripped := parseModelCacheControl(in)
		if forceFresh || maxAgeMS != 0 || hasMaxAge || reuseOnly || stripped != in {
			t.Fatalf("got (%v, %d, %v, %v, %q), want (false, 0, false, false, %q)",
				forceFresh, maxAgeMS, hasMaxAge, reuseOnly, stripped, in)
		}
	})

	t.Run("invalid json returns args unchanged", func(t *testing.T) {
		in := `{"_fak_max_age_ms":`
		forceFresh, maxAgeMS, hasMaxAge, reuseOnly, stripped := parseModelCacheControl(in)
		if forceFresh || maxAgeMS != 0 || hasMaxAge || reuseOnly || stripped != in {
			t.Fatalf("got (%v, %d, %v, %v, %q), want unchanged",
				forceFresh, maxAgeMS, hasMaxAge, reuseOnly, stripped)
		}
	})

	t.Run("_fak_fresh preserved and sets forceFresh", func(t *testing.T) {
		in := `{"id":"x","_fak_fresh":true}`
		forceFresh, maxAgeMS, hasMaxAge, reuseOnly, stripped := parseModelCacheControl(in)
		if !forceFresh || maxAgeMS != 0 || hasMaxAge || reuseOnly || stripped != in {
			t.Fatalf("got (%v, %d, %v, %v, %q)", forceFresh, maxAgeMS, hasMaxAge, reuseOnly, stripped)
		}
	})

	t.Run("_fak_max_age_ms zero sets forceFresh and strips key", func(t *testing.T) {
		in := `{"id":"x","_fak_max_age_ms":0}`
		forceFresh, maxAgeMS, hasMaxAge, reuseOnly, stripped := parseModelCacheControl(in)
		if !forceFresh || maxAgeMS != 0 || !hasMaxAge || reuseOnly {
			t.Fatalf("got (%v, %d, %v, %v)", forceFresh, maxAgeMS, hasMaxAge, reuseOnly)
		}
		if stripped != `{"id":"x"}` {
			t.Fatalf("stripped=%q, want %q", stripped, `{"id":"x"}`)
		}
	})

	t.Run("_fak_max_age_ms positive sets bound and strips key", func(t *testing.T) {
		in := `{"id":"x","_fak_max_age_ms":5000}`
		forceFresh, maxAgeMS, hasMaxAge, reuseOnly, stripped := parseModelCacheControl(in)
		if forceFresh || maxAgeMS != 5000 || !hasMaxAge || reuseOnly {
			t.Fatalf("got (%v, %d, %v, %v)", forceFresh, maxAgeMS, hasMaxAge, reuseOnly)
		}
		if stripped != `{"id":"x"}` {
			t.Fatalf("stripped=%q, want %q", stripped, `{"id":"x"}`)
		}
	})

	t.Run("negative or invalid _fak_max_age_ms ignored fail-safe and stripped", func(t *testing.T) {
		for _, in := range []string{
			`{"id":"x","_fak_max_age_ms":-5}`,
			`{"id":"x","_fak_max_age_ms":"5000"}`,
			`{"id":"x","_fak_max_age_ms":1.5}`,
			`{"id":"x","_fak_max_age_ms":null}`,
		} {
			forceFresh, maxAgeMS, hasMaxAge, reuseOnly, stripped := parseModelCacheControl(in)
			if forceFresh || maxAgeMS != 0 || hasMaxAge || reuseOnly {
				t.Fatalf("%s: got (%v, %d, %v, %v), want all false/0", in, forceFresh, maxAgeMS, hasMaxAge, reuseOnly)
			}
			if stripped != `{"id":"x"}` {
				t.Fatalf("%s: stripped=%q, want %q", in, stripped, `{"id":"x"}`)
			}
		}
	})

	t.Run("_fak_reuse_only true sets reuseOnly and strips key", func(t *testing.T) {
		in := `{"id":"x","_fak_reuse_only":true}`
		forceFresh, maxAgeMS, hasMaxAge, reuseOnly, stripped := parseModelCacheControl(in)
		if forceFresh || maxAgeMS != 0 || hasMaxAge || !reuseOnly {
			t.Fatalf("got (%v, %d, %v, %v)", forceFresh, maxAgeMS, hasMaxAge, reuseOnly)
		}
		if stripped != `{"id":"x"}` {
			t.Fatalf("stripped=%q, want %q", stripped, `{"id":"x"}`)
		}
	})

	t.Run("combined markers strip _fak_max_age_ms and _fak_reuse_only while keeping _fak_fresh", func(t *testing.T) {
		in := `{"id":"x","_fak_fresh":true,"_fak_max_age_ms":2500,"_fak_reuse_only":true}`
		forceFresh, maxAgeMS, hasMaxAge, reuseOnly, stripped := parseModelCacheControl(in)
		if !forceFresh || maxAgeMS != 2500 || !hasMaxAge || !reuseOnly {
			t.Fatalf("got (%v, %d, %v, %v)", forceFresh, maxAgeMS, hasMaxAge, reuseOnly)
		}
		var m map[string]any
		if err := json.Unmarshal([]byte(stripped), &m); err != nil {
			t.Fatalf("unmarshal stripped: %v", err)
		}
		if _, ok := m[fakMaxAgeMSMarker]; ok {
			t.Fatalf("stripped still contains %s: %s", fakMaxAgeMSMarker, stripped)
		}
		if _, ok := m[fakReuseOnlyMarker]; ok {
			t.Fatalf("stripped still contains %s: %s", fakReuseOnlyMarker, stripped)
		}
		if v, ok := m[fakFreshMarker]; !ok || v != true {
			t.Fatalf("stripped missing %s=true: %s", fakFreshMarker, stripped)
		}
	})
}

func TestModelCacheControl_HitExceedsModelMaxAge(t *testing.T) {
	if hitExceedsModelMaxAge(map[string]string{"age_ms": "99999"}, 1000, false) {
		t.Fatal("hasMaxAge=false must return false")
	}
	if hitExceedsModelMaxAge(map[string]string{"tier": "1"}, 1000, true) {
		t.Fatal("missing age_ms must return false")
	}
	if hitExceedsModelMaxAge(map[string]string{"age_ms": "1000"}, 1000, true) {
		t.Fatal("equal age_ms == maxAgeMS must return false")
	}
	if !hitExceedsModelMaxAge(map[string]string{"age_ms": "1001"}, 1000, true) {
		t.Fatal("age_ms > maxAgeMS must return true")
	}
}

func TestModelCacheControl_MaxAge(t *testing.T) {
	const tool = "get_quote"
	srv := newAgeStubServer(t, ageStubFastPath{tool: tool, body: `{"px":42}`, ageMs: "5000"}) // 5s old
	ctx := context.Background()

	// 1. Fresh vDSO hit within _fak_max_age_ms bound serves inline.
	kept, adjs, dropped, servedText, servedHits := srv.adjudicateProposedServed(ctx, []agent.ToolCall{
		{ID: "c1", Type: "function", Function: agent.Func{Name: tool, Arguments: `{"id":"x","_fak_max_age_ms":10000}`}},
	}, "trace-maxage-within")
	if len(kept) != 0 || dropped != 0 || servedHits != 1 || !strings.Contains(servedText, "served from cache") {
		t.Fatalf("within bound: kept=%d dropped=%d servedHits=%d text=%q adjs=%+v", len(kept), dropped, servedHits, servedText, adjs)
	}

	// 2. Stale vDSO hit exceeding _fak_max_age_ms misses and passes through with stripped args.
	kept, adjs, dropped, servedText, servedHits = srv.adjudicateProposedServed(ctx, []agent.ToolCall{
		{ID: "c2", Type: "function", Function: agent.Func{Name: tool, Arguments: `{"id":"x","_fak_max_age_ms":1000}`}},
	}, "trace-maxage-exceeded")
	if len(kept) != 1 || dropped != 0 || servedHits != 0 || servedText != "" {
		t.Fatalf("exceeded bound: kept=%d dropped=%d servedHits=%d text=%q adjs=%+v", len(kept), dropped, servedHits, servedText, adjs)
	}
	if kept[0].Function.Arguments != `{"id":"x"}` {
		t.Fatalf("exceeded bound: kept args=%q, want stripped %q", kept[0].Function.Arguments, `{"id":"x"}`)
	}

	// 3. Operator SetToolMaxAge stricter than _fak_max_age_ms still rejects.
	srv.SetToolMaxAge(tool, 2*time.Second)
	kept, adjs, dropped, servedText, servedHits = srv.adjudicateProposedServed(ctx, []agent.ToolCall{
		{ID: "c3", Type: "function", Function: agent.Func{Name: tool, Arguments: `{"id":"x","_fak_max_age_ms":60000}`}},
	}, "trace-maxage-operator-wins")
	if len(kept) != 1 || dropped != 0 || servedHits != 0 || servedText != "" {
		t.Fatalf("operator ceiling: kept=%d dropped=%d servedHits=%d text=%q adjs=%+v", len(kept), dropped, servedHits, servedText, adjs)
	}
	if kept[0].Function.Arguments != `{"id":"x"}` {
		t.Fatalf("operator ceiling: kept args=%q, want stripped %q", kept[0].Function.Arguments, `{"id":"x"}`)
	}
}

func TestModelCacheControl_ReuseOnly(t *testing.T) {
	ctx := context.Background()

	t.Run("cache hit serves inline", func(t *testing.T) {
		const tool = "get_quote"
		srv := newAgeStubServer(t, ageStubFastPath{tool: tool, body: `{"px":42}`, ageMs: "5000"})
		kept, adjs, dropped, servedText, servedHits := srv.adjudicateProposedServed(ctx, []agent.ToolCall{
			{ID: "r1", Type: "function", Function: agent.Func{Name: tool, Arguments: `{"id":"x","_fak_reuse_only":true}`}},
		}, "trace-reuse-hit")
		if len(kept) != 0 || dropped != 0 || servedHits != 1 || !strings.Contains(servedText, "served from cache") {
			t.Fatalf("reuse hit: kept=%d dropped=%d servedHits=%d text=%q adjs=%+v", len(kept), dropped, servedHits, servedText, adjs)
		}
	})

	t.Run("unarmed cache miss skips live execution", func(t *testing.T) {
		srv := newBareReuseServer(t)
		kept, adjs, dropped, servedText, servedHits := srv.adjudicateProposedServed(ctx, []agent.ToolCall{
			{ID: "r2", Type: "function", Function: agent.Func{Name: "get_quote", Arguments: `{"id":"x","_fak_reuse_only":true}`}},
		}, "trace-reuse-unarmed")
		if len(kept) != 0 || servedHits != 0 || dropped != 0 {
			t.Fatalf("unarmed miss: kept=%d servedHits=%d dropped=%d", len(kept), servedHits, dropped)
		}
		if len(adjs) != 1 || adjs[0].Verdict.Reason != "CACHE_MISS_SKIPPED" || adjs[0].Verdict.By != "model-cache-control" || !adjs[0].Admitted {
			t.Fatalf("unarmed miss: adjs=%+v", adjs)
		}
		if !strings.Contains(servedText, "CACHE_MISS_SKIPPED") && !strings.Contains(servedText, "skipped fresh execution") {
			t.Fatalf("unarmed miss: unexpected servedText=%q", servedText)
		}
	})

	t.Run("write-shaped tool miss skips live execution", func(t *testing.T) {
		srv := newAgeStubServer(t, ageStubFastPath{tool: "get_quote", body: `{"px":42}`, ageMs: "5000"})
		kept, adjs, dropped, servedText, servedHits := srv.adjudicateProposedServed(ctx, []agent.ToolCall{
			{ID: "r3", Type: "function", Function: agent.Func{Name: "write_file", Arguments: `{"path":"a.go","_fak_reuse_only":true}`}},
		}, "trace-reuse-write")
		if len(kept) != 0 || servedHits != 0 || dropped != 0 {
			t.Fatalf("write-shaped miss: kept=%d servedHits=%d dropped=%d", len(kept), servedHits, dropped)
		}
		if len(adjs) != 1 || adjs[0].Verdict.Reason != "CACHE_MISS_SKIPPED" || !strings.Contains(servedText, "skipped fresh execution") {
			t.Fatalf("write-shaped miss: adjs=%+v text=%q", adjs, servedText)
		}
	})

	t.Run("over-age miss skips live execution", func(t *testing.T) {
		const tool = "get_quote"
		srv := newAgeStubServer(t, ageStubFastPath{tool: tool, body: `{"px":42}`, ageMs: "5000"})
		kept, adjs, dropped, servedText, servedHits := srv.adjudicateProposedServed(ctx, []agent.ToolCall{
			{ID: "r4", Type: "function", Function: agent.Func{Name: tool, Arguments: `{"id":"x","_fak_max_age_ms":1000,"_fak_reuse_only":true}`}},
		}, "trace-reuse-overage")
		if len(kept) != 0 || servedHits != 0 || dropped != 0 {
			t.Fatalf("over-age miss: kept=%d servedHits=%d dropped=%d", len(kept), servedHits, dropped)
		}
		if len(adjs) != 1 || adjs[0].Verdict.Reason != "CACHE_MISS_SKIPPED" || !strings.Contains(servedText, "skipped fresh execution") {
			t.Fatalf("over-age miss: adjs=%+v text=%q", adjs, servedText)
		}
	})
}
