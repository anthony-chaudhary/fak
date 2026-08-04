package gateway

// toolproc_reuse_test.go — the ACCEPTANCE WITNESS for #5119: the three safety
// properties of the live toolproc reuse seam, exercised through the REAL gateway
// loop (adjudicateProposedServed → admitInboundResults), not a model of it.
//
// What each test proves, mapped to the issue's acceptance gate:
//
//	"an immutable skill re-read is served locally (receipt shows the digest key)"
//	  → TestToolprocReuseServesImmutableRereadOnLiveSeam
//	"a mutation is never served stale"
//	  → TestToolprocReuseInvalidatesOnMutation
//	"a mutable status poll coalesces only inside its freshness window"
//	  → TestToolprocReuseCoalescesQueryOnlyInsideFreshnessWindow
//	"a write/unknown is never reused"
//	  → TestToolprocReuseNeverServesWriteOrUnknown
//	"…and arming it is opt-in, so the pre-#5119 path is byte-identical"
//	  → TestToolprocReuseUnarmedIsInert (and TestServedInlineGuardTrace next door,
//	    whose measured 0 served-inline on native names still holds unarmed)
//
// The digest-witness refusal (TestToolprocReuseRefusesReadWithoutDigestWitness) is
// the property that makes the first one SOUND rather than merely fast: without a
// current content digest there is no key, and without a key nothing is served.

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/toolproc"
)

// newToolprocReuseServer builds a gateway with the real result-side admission stack
// and the toolproc reuse seam ARMED over root. VDSOProxyFill stays OFF on purpose:
// every serve below must be attributable to THIS seam, not to the vDSO probe that
// shares adjudicateProposedServed.
func newToolprocReuseServer(t *testing.T, root string, cfg toolproc.ArmedConfig) *Server {
	t.Helper()
	srv := newBareReuseServer(t)
	srv.SetToolprocReuse(cfg, toolproc.FileDigest(root))
	t.Cleanup(srv.DisarmToolprocReuse)
	return srv
}

// newBareReuseServer builds the same gateway with NOTHING armed — the default
// posture, and the negative control for every serve assertion below.
func newBareReuseServer(t *testing.T) *Server {
	t.Helper()
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, allowAllAdj{})
	abi.RegisterResultAdmitter(10, ctxmmu.New())
	// ResetForTest clears reg.reasons, and the REUSE_* vocabulary is registered by
	// toolprocgate's package INIT — which already ran, and does not run again. Without
	// this restore a served hit still serves, but cites the REASON_<n> forward-compat
	// spelling instead of its stable token: the same drift any gateway test that resets
	// the registry inflicts on every init-registered vocabulary. Production never
	// resets, so this is the harness paying back what it took, not a seam workaround.
	for _, pr := range toolproc.ReuseReasonPairs() {
		abi.RegisterReason(pr.Code, pr.Name)
	}
	srv, err := New(Config{EngineID: "test", Model: "test"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(srv.Close)
	return srv
}

// reuseTurn drives ONE proposed tool call through the real seam and, when the call
// survived to the wire, feeds its result back exactly as the proxy flow does — the
// serve half and the deposit half of the same loop, in the order messages.go runs
// them. It returns whether the call was answered locally and the served line.
func reuseTurn(t *testing.T, srv *Server, ctx context.Context, trace, id, tool, args, result string) (served bool, line string) {
	t.Helper()
	call := agent.ToolCall{ID: id, Type: "function", Function: agent.Func{Name: tool, Arguments: args}}
	kept, adjs, dropped, servedText, servedHits := srv.adjudicateProposedServed(ctx, []agent.ToolCall{call}, trace)
	if dropped != 0 {
		t.Fatalf("%s %s: dropped=%d, want 0 (allowAllAdj must admit)", tool, args, dropped)
	}
	if servedHits > 0 {
		if len(kept) != 0 {
			t.Fatalf("%s %s: served inline but still kept %d calls — the buckets must be disjoint", tool, args, len(kept))
		}
		if len(adjs) != 1 || adjs[0].Verdict.Reason != "SERVED_INLINE" || adjs[0].Verdict.By != ReuseServedByVerdict {
			t.Fatalf("%s %s: served hit must cite SERVED_INLINE by %q, got %+v", tool, args, ReuseServedByVerdict, adjs)
		}
		return true, servedText
	}
	if len(kept) != 1 {
		t.Fatalf("%s %s: miss must keep the call, got kept=%d", tool, args, len(kept))
	}
	// The client ran it: hand the result back through the ADMISSION path, which is
	// where the deposit half lives.
	inbound := []agent.Message{
		{Role: agent.RoleUser, Content: "go"},
		{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{call}},
		{Role: agent.RoleTool, ToolCallID: id, Name: tool, Content: result},
	}
	if _, err := srv.admitInboundResults(ctx, inbound, nil, trace); err != nil {
		t.Fatalf("admitInboundResults: %v", err)
	}
	return false, ""
}

// writeReuseFile writes one file under a fresh temp dir and returns (dir, name).
func writeReuseFile(t *testing.T, name, content string) (string, string) {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o600); err != nil {
		t.Fatalf("write %s: %v", name, err)
	}
	return dir, name
}

// TestToolprocReuseServesImmutableRereadOnLiveSeam is the acceptance gate's first
// clause: a repeated immutable read is answered LOCALLY on the live seam, and the
// receipt names the (resolved path + content digest) key it was served under.
func TestToolprocReuseServesImmutableRereadOnLiveSeam(t *testing.T) {
	dir, name := writeReuseFile(t, "skill.md", "# the skill body that must not be re-fetched\n")
	srv := newToolprocReuseServer(t, dir, toolproc.ArmedConfig{})
	ctx := WithPrincipal(context.Background(), "tenantReuse")
	args := `{"file_path":"` + name + `"}`
	const body = `{"content":"# the skill body that must not be re-fetched"}`

	// Turn 1: nothing is cached, so the read must reach the client and its result
	// must be deposited.
	if served, _ := reuseTurn(t, srv, ctx, "tr", "c1", "Read", args, body); served {
		t.Fatal("first read must NOT be served: nothing has been fetched yet")
	}
	// Turn 2: the identical re-read is answered locally.
	served, line := reuseTurn(t, srv, ctx, "tr", "c2", "Read", args, body)
	if !served {
		t.Fatal("re-read of an unchanged file must be served locally")
	}
	if !strings.Contains(line, "must not be re-fetched") {
		t.Fatalf("served line must carry the cached body, got %q", line)
	}

	// The RECEIPT: the identity is the digest key, and the verdict/provenance are
	// the registered reuse vocabulary — not free text.
	_, meta, ok := srv.reuseServe(ctx, "Read", args)
	if !ok {
		t.Fatal("reuseServe must report the same hit the live seam took")
	}
	wantDigest := toolproc.FileDigest(dir)(name)
	if wantDigest == "" {
		t.Fatal("FileDigest must witness the temp file")
	}
	if got, want := meta[ReuseIdentityMetaKey], "read:"+name+"@"+wantDigest; got != want {
		t.Errorf("reuse_identity = %q, want the digest key %q", got, want)
	}
	if got := meta["reuse_reason"]; got != toolproc.ReasonReuseKeyedHitName {
		t.Errorf("reuse_reason = %q, want %q", got, toolproc.ReasonReuseKeyedHitName)
	}
	if got := meta["reuse_source"]; got != string(toolproc.SourceImmutable) {
		t.Errorf("reuse_source = %q, want %q", got, toolproc.SourceImmutable)
	}
	if meta[ReuseSavedBytesMetaKey] == "" {
		t.Error("a served hit must report the bytes it did not re-fetch")
	}
	// A digest-keyed hit is keyed on the content as it is RIGHT NOW, so it is not
	// stale by any amount and must carry no age clause.
	if _, hasAge := meta["age_ms"]; hasAge {
		t.Errorf("an immutable digest-keyed hit must carry no age, got %q", meta["age_ms"])
	}
}

// TestToolprocReuseInvalidatesOnMutation: a file mutated between two reads forms a
// NEW key, so the stale body is never served — the invalidation-after-mutation
// contract, on the live seam.
func TestToolprocReuseInvalidatesOnMutation(t *testing.T) {
	dir, name := writeReuseFile(t, "skill.md", "original\n")
	srv := newToolprocReuseServer(t, dir, toolproc.ArmedConfig{})
	ctx := WithPrincipal(context.Background(), "tenantReuse")
	args := `{"file_path":"` + name + `"}`

	reuseTurn(t, srv, ctx, "tr", "c1", "Read", args, `{"content":"original"}`)
	if served, _ := reuseTurn(t, srv, ctx, "tr", "c2", "Read", args, `{"content":"original"}`); !served {
		t.Fatal("unchanged re-read must serve (control for the mutation below)")
	}

	if err := os.WriteFile(filepath.Join(dir, name), []byte("MUTATED\n"), 0o600); err != nil {
		t.Fatalf("mutate: %v", err)
	}
	served, line := reuseTurn(t, srv, ctx, "tr", "c3", "Read", args, `{"content":"MUTATED"}`)
	if served {
		t.Fatalf("a read after a mutation must NOT be served from the stale entry, got %q", line)
	}
	// And the fresh content, once deposited under the new key, serves again.
	if served, line := reuseTurn(t, srv, ctx, "tr", "c4", "Read", args, `{"content":"MUTATED"}`); !served || !strings.Contains(line, "MUTATED") {
		t.Fatalf("post-mutation re-read must serve the NEW content, served=%v line=%q", served, line)
	}
}

// TestToolprocReuseCoalescesQueryOnlyInsideFreshnessWindow: a registered mutable
// status command coalesces inside its window and STOPS at the boundary, and the hit
// exposes its stale-age and freshness-window provenance.
func TestToolprocReuseCoalescesQueryOnlyInsideFreshnessWindow(t *testing.T) {
	dir := t.TempDir()
	const args = `{"command":"git status --porcelain"}`
	const body = `{"stdout":" M internal/gateway/toolproc_reuse.go"}`

	t.Run("inside the window", func(t *testing.T) {
		srv := newToolprocReuseServer(t, dir, toolproc.ArmedConfig{
			CoalesceQueries: true,
			Repeat:          toolproc.RepeatConfig{DefaultFreshnessMS: 60_000},
		})
		ctx := WithPrincipal(context.Background(), "tenantReuse")
		reuseTurn(t, srv, ctx, "tr", "q1", "Bash", args, body)
		served, line := reuseTurn(t, srv, ctx, "tr", "q2", "Bash", args, body)
		if !served {
			t.Fatal("a status poll inside its freshness window must coalesce")
		}
		if !strings.Contains(line, "toolproc_reuse.go") {
			t.Fatalf("coalesced line must carry the cached stdout, got %q", line)
		}
		_, meta, ok := srv.reuseServe(ctx, "Bash", args)
		if !ok {
			t.Fatal("reuseServe must report the same coalesced hit")
		}
		if got := meta["reuse_source"]; got != string(toolproc.SourceFreshness) {
			t.Errorf("reuse_source = %q, want %q — a window hit must not read as live", got, toolproc.SourceFreshness)
		}
		if got := meta["reuse_reason"]; got != toolproc.ReasonReuseFreshnessHitName {
			t.Errorf("reuse_reason = %q, want %q", got, toolproc.ReasonReuseFreshnessHitName)
		}
		if got := meta[ReuseIdentityMetaKey]; !strings.HasPrefix(got, "query:git status") {
			t.Errorf("reuse_identity = %q, want a query: identity", got)
		}
	})

	t.Run("past the window", func(t *testing.T) {
		srv := newToolprocReuseServer(t, dir, toolproc.ArmedConfig{
			CoalesceQueries: true,
			Repeat:          toolproc.RepeatConfig{DefaultFreshnessMS: 1},
		})
		ctx := WithPrincipal(context.Background(), "tenantReuse")
		reuseTurn(t, srv, ctx, "tr", "q1", "Bash", args, body)
		time.Sleep(30 * time.Millisecond) // 30x the 1ms window
		if served, line := reuseTurn(t, srv, ctx, "tr", "q2", "Bash", args, body); served {
			t.Fatalf("a status poll PAST its freshness window must run fresh, got %q", line)
		}
	})

	t.Run("coalescing not opted in", func(t *testing.T) {
		srv := newToolprocReuseServer(t, dir, toolproc.ArmedConfig{
			Repeat: toolproc.RepeatConfig{DefaultFreshnessMS: 60_000},
		})
		ctx := WithPrincipal(context.Background(), "tenantReuse")
		reuseTurn(t, srv, ctx, "tr", "q1", "Bash", args, body)
		if served, _ := reuseTurn(t, srv, ctx, "tr", "q2", "Bash", args, body); served {
			t.Fatal("mutable-status coalescing is OPT-IN: unopted, a poll must never be answered locally")
		}
	})
}

// TestToolprocReuseNeverServesWriteOrUnknown: the fail-closed half. A registered
// write, an unregistered command, and a write-SHAPED tool name are each refused
// twice over — never retained on the deposit side, never answered on the serve side.
func TestToolprocReuseNeverServesWriteOrUnknown(t *testing.T) {
	dir, name := writeReuseFile(t, "notes.md", "notes\n")
	cases := []struct {
		what string
		tool string
		args string
	}{
		{"registered write", "Bash", `{"command":"git commit -m x"}`},
		{"unregistered command", "Bash", `{"command":"curl https://example.invalid"}`},
		{"write-shaped tool name over a read path", "Write", `{"file_path":"` + name + `"}`},
		{"write-shaped tool name over a read command", "Edit", `{"command":"cat ` + name + `"}`},
	}
	for _, tc := range cases {
		t.Run(tc.what, func(t *testing.T) {
			srv := newToolprocReuseServer(t, dir, toolproc.ArmedConfig{
				CoalesceQueries: true,
				Repeat:          toolproc.RepeatConfig{DefaultFreshnessMS: 60_000},
			})
			ctx := WithPrincipal(context.Background(), "tenantReuse")
			for i, id := range []string{"w1", "w2", "w3"} {
				if served, line := reuseTurn(t, srv, ctx, "tr", id, tc.tool, tc.args, `{"ok":true}`); served {
					t.Fatalf("call %d: %s must NEVER be reused, got %q", i, tc.what, line)
				}
			}
		})
	}
}

// TestToolprocReuseRefusesReadWithoutDigestWitness: with no way to witness the
// target's CURRENT content, there is no key — and toolproc's own path-only fallback
// (right for offline analytics of a finished rollout) must not be allowed to serve
// a live read whose file may have changed.
func TestToolprocReuseRefusesReadWithoutDigestWitness(t *testing.T) {
	_, name := writeReuseFile(t, "skill.md", "body\n")
	srv := newBareReuseServer(t)
	srv.SetToolprocReuse(toolproc.ArmedConfig{}, nil) // armed, but no digest witness
	t.Cleanup(srv.DisarmToolprocReuse)
	ctx := WithPrincipal(context.Background(), "tenantReuse")
	args := `{"file_path":"` + name + `"}`

	for _, id := range []string{"n1", "n2", "n3"} {
		if served, _ := reuseTurn(t, srv, ctx, "tr", id, "Read", args, `{"content":"body"}`); served {
			t.Fatal("a read with no witnessed digest must never be served")
		}
	}

	// Same call, but under a root that CAN witness the file: the only difference is
	// the digest, which is what makes this a witness and not an availability quirk.
	dir2, name2 := writeReuseFile(t, "skill.md", "body\n")
	srv2 := newToolprocReuseServer(t, dir2, toolproc.ArmedConfig{})
	args2 := `{"file_path":"` + name2 + `"}`
	reuseTurn(t, srv2, ctx, "tr", "y1", "Read", args2, `{"content":"body"}`)
	if served, _ := reuseTurn(t, srv2, ctx, "tr", "y2", "Read", args2, `{"content":"body"}`); !served {
		t.Fatal("control: with a digest witness the same re-read MUST serve")
	}
}

// TestToolprocReuseUnarmedIsInert: the default. An unarmed Server never serves,
// never retains, and its adjudication path is the pre-#5119 one — which is what
// makes arming this a per-operator decision rather than a fleet-wide flip.
func TestToolprocReuseUnarmedIsInert(t *testing.T) {
	_, name := writeReuseFile(t, "skill.md", "body\n")
	srv := newBareReuseServer(t)
	ctx := WithPrincipal(context.Background(), "tenantReuse")
	args := `{"file_path":"` + name + `"}`

	for _, id := range []string{"u1", "u2", "u3"} {
		if served, _ := reuseTurn(t, srv, ctx, "tr", id, "Read", args, `{"content":"body"}`); served {
			t.Fatal("an unarmed gateway must never serve a repeat")
		}
	}
	if _, _, ok := srv.reuseServe(ctx, "Read", args); ok {
		t.Error("reuseServe on an unarmed Server must miss")
	}
	srv.reuseOffer("Read", args, `{"content":"body"}`) // must not panic
	if sm := srv.reuseSeamOf(); sm != nil {
		t.Errorf("an unarmed Server must have no seam, got %+v", sm)
	}

	// Disarming an armed Server returns it to exactly this state, dropping the bytes.
	dir2, name2 := writeReuseFile(t, "skill.md", "body\n")
	srv2 := newToolprocReuseServer(t, dir2, toolproc.ArmedConfig{})
	args2 := `{"file_path":"` + name2 + `"}`
	reuseTurn(t, srv2, ctx, "tr", "d1", "Read", args2, `{"content":"body"}`)
	srv2.DisarmToolprocReuse()
	if served, _ := reuseTurn(t, srv2, ctx, "tr", "d2", "Read", args2, `{"content":"body"}`); served {
		t.Fatal("a disarmed gateway must not serve bytes it retained while armed")
	}
}

// TestToolprocReuseHonorsForceFresh: the model's _fak_fresh escape hatch must beat
// this cache too, or "force a fresh read" would be a lie for exactly the calls this
// seam is best at serving.
func TestToolprocReuseHonorsForceFresh(t *testing.T) {
	dir, name := writeReuseFile(t, "skill.md", "body\n")
	srv := newToolprocReuseServer(t, dir, toolproc.ArmedConfig{})
	ctx := WithPrincipal(context.Background(), "tenantReuse")
	args := `{"file_path":"` + name + `"}`

	reuseTurn(t, srv, ctx, "tr", "f1", "Read", args, `{"content":"body"}`)
	if served, _ := reuseTurn(t, srv, ctx, "tr", "f2", "Read", args, `{"content":"body"}`); !served {
		t.Fatal("control: the plain re-read must serve")
	}
	fresh := `{"file_path":"` + name + `","` + fakFreshMarker + `":true}`
	if served, _ := reuseTurn(t, srv, ctx, "tr", "f3", "Read", fresh, `{"content":"body"}`); served {
		t.Fatal("a call carrying _fak_fresh must bypass the toolproc reuse cache")
	}
}
