package main

// serve_context_plane_test.go — the regression lock for the serving router's
// context-plane posture resolution (the 2026-11-10 hosting-simplification flip).
// The serve door resolves CtxPaging and ChatToolResultElide OFF by default; the
// per-lever serving flags (--ctx-view-budget, --compact-history-budget,
// --elide-result-bytes, --elide-stale-reads, --vcache-anchor, --defer-cold-tools)
// default to their raw off values. Precedence mirrors the tool guard: an
// explicitly passed flag wins over the env, an explicit env wins over the flag
// default, and an unparsable value fails loud — a context rewriter never
// silently resolves.

import (
	"flag"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

func TestResolveServeCtxPagingPrecedence(t *testing.T) {
	getenv := func(k string) string { return "" }

	// Default (no flag, no env): the raw serving-router posture.
	got, err := resolveServeCtxPaging(gateway.DefaultServeCtxPaging, nil, getenv)
	if err != nil || got != gateway.CtxPagingOff {
		t.Fatalf("default = (%q,%v), want (off,nil)", got, err)
	}

	// Explicit flag wins over everything, including its own default value.
	explicit := map[string]bool{"ctx-paging": true}
	got, err = resolveServeCtxPaging("auto", explicit, func(k string) string { return "off" })
	if err != nil || got != gateway.CtxPagingAuto {
		t.Fatalf("explicit flag must win over env: got (%q,%v), want (auto,nil)", got, err)
	}

	// An explicit env wins over the flag DEFAULT (flag not explicitly passed).
	got, err = resolveServeCtxPaging(gateway.DefaultServeCtxPaging, nil, func(k string) string { return "on" })
	if err != nil || got != gateway.CtxPagingOn {
		t.Fatalf("env must win over the flag default: got (%q,%v), want (on,nil)", got, err)
	}

	// An unparsable env fails loud.
	if _, err := resolveServeCtxPaging(gateway.DefaultServeCtxPaging, nil, func(k string) string { return "perhaps" }); err == nil || !strings.Contains(err.Error(), "FAK_CTX_PAGING") {
		t.Fatalf("unparsable FAK_CTX_PAGING must fail loud naming the env, got %v", err)
	}

	// An unparsable explicit flag fails loud.
	if _, err := resolveServeCtxPaging("perhaps", explicit, getenv); err == nil {
		t.Fatal("an unparsable explicit --ctx-paging must fail loud")
	}
}

func TestResolveServeChatToolResultElidePrecedence(t *testing.T) {
	getenv := func(k string) string { return "" }

	// Default (no flag, no env): the raw serving-router posture.
	got, err := resolveServeChatToolResultElide(false, nil, getenv)
	if err != nil || got != gateway.ChatToolResultElideOff {
		t.Fatalf("default = (%q,%v), want (off,nil)", got, err)
	}

	// Explicit flag wins over everything.
	explicit := map[string]bool{"chat-elide": true}
	got, err = resolveServeChatToolResultElide(true, explicit, func(k string) string { return "off" })
	if err != nil || got != gateway.ChatToolResultElideOn {
		t.Fatalf("explicit flag must win over env: got (%q,%v), want (on,nil)", got, err)
	}
	got, err = resolveServeChatToolResultElide(false, explicit, func(k string) string { return "on" })
	if err != nil || got != gateway.ChatToolResultElideOff {
		t.Fatalf("explicit --chat-elide=false must win over env: got (%q,%v), want (off,nil)", got, err)
	}

	// An explicit env wins over the flag DEFAULT.
	got, err = resolveServeChatToolResultElide(false, nil, func(k string) string { return "on" })
	if err != nil || got != gateway.ChatToolResultElideOn {
		t.Fatalf("env must win over the flag default: got (%q,%v), want (on,nil)", got, err)
	}

	// An unparsable env fails loud.
	if _, err := resolveServeChatToolResultElide(false, nil, func(k string) string { return "perhaps" }); err == nil || !strings.Contains(err.Error(), "FAK_CHAT_ELIDE") {
		t.Fatalf("unparsable FAK_CHAT_ELIDE must fail loud naming the env, got %v", err)
	}
}

// TestServeContextPlaneFlagsWiredToPostureConsts pins the live flag registrations:
// --ctx-paging defaults to gateway.DefaultServeCtxPaging and --chat-elide derives its
// bool from gateway.DefaultServeChatToolResultElide, so the posture consts stay the
// single authoritative default for the serving router.
func TestServeContextPlaneFlagsWiredToPostureConsts(t *testing.T) {
	serve := readEntrypoint(t, "serve.go")
	if !strings.Contains(serve, `fs.String("ctx-paging", gateway.DefaultServeCtxPaging`) {
		t.Errorf("serve.go must wire --ctx-paging to gateway.DefaultServeCtxPaging (the raw serving-router posture)")
	}
	if !strings.Contains(serve, `fs.Bool("chat-elide", gateway.DefaultServeChatToolResultElide`) {
		t.Errorf("serve.go must derive --chat-elide from gateway.DefaultServeChatToolResultElide (the raw serving-router posture)")
	}
	if !strings.Contains(serve, "CtxPaging:                    ctxPagingMode,") && !strings.Contains(strings.Join(strings.Fields(serve), " "), "CtxPaging: ctxPagingMode,") {
		t.Errorf("serve.go must pass the resolved ctx-paging posture into gateway.Config.CtxPaging")
	}
	if !strings.Contains(strings.Join(strings.Fields(serve), " "), "ChatToolResultElide: chatElideMode,") {
		t.Errorf("serve.go must pass the resolved chat-elide posture into gateway.Config.ChatToolResultElide")
	}
}

// TestServeSaverFlagsParseRawDefaults is the behavioral half: FlagSets declared exactly
// as the serve door declares them parse to the RAW values with no arguments — the
// serving router arms no rewriter by default — and each explicit value still wins.
func TestServeSaverFlagsParseRawDefaults(t *testing.T) {
	fs := flag.NewFlagSet("t", flag.ContinueOnError)
	ctxView := fs.Int("ctx-view-budget", gateway.DefaultServeCtxViewBudget, "")
	compact := fs.Int("compact-history-budget", gateway.DefaultServeCompactHistoryBudget, "")
	elideBytes := fs.Int("elide-result-bytes", gateway.DefaultServeElideResultBytes, "")
	elideStale := fs.Bool("elide-stale-reads", gateway.DefaultServeElideStaleReads, "")
	vcache := fs.Bool("vcache-anchor", gateway.DefaultServeVCacheAnchor, "")
	deferCold := fs.Bool("defer-cold-tools", gateway.DefaultServeDeferColdTools, "")
	paging := fs.String("ctx-paging", gateway.DefaultServeCtxPaging, "")
	chatElide := fs.Bool("chat-elide", gateway.DefaultServeChatToolResultElide == gateway.ChatToolResultElideOn, "")
	if err := fs.Parse(nil); err != nil {
		t.Fatalf("parse empty: %v", err)
	}
	if *ctxView != 0 || *compact != 0 || *elideBytes != 0 {
		t.Fatalf("serve saver budgets must parse to the raw 0 defaults, got ctxview=%d compact=%d elide=%d", *ctxView, *compact, *elideBytes)
	}
	if *elideStale || *vcache || *deferCold || *chatElide {
		t.Fatalf("serve saver bools must parse to the raw false defaults, got elidestale=%v vcache=%v defercold=%v chatelide=%v", *elideStale, *vcache, *deferCold, *chatElide)
	}
	if *paging != gateway.CtxPagingOff {
		t.Fatalf("serve --ctx-paging must parse to off, got %q", *paging)
	}

	// Every lever still re-arms by name on the serve door.
	fs2 := flag.NewFlagSet("t", flag.ContinueOnError)
	ctxView2 := fs2.Int("ctx-view-budget", gateway.DefaultServeCtxViewBudget, "")
	compact2 := fs2.Int("compact-history-budget", gateway.DefaultServeCompactHistoryBudget, "")
	elideStale2 := fs2.Bool("elide-stale-reads", gateway.DefaultServeElideStaleReads, "")
	paging2 := fs2.String("ctx-paging", gateway.DefaultServeCtxPaging, "")
	if err := fs2.Parse([]string{"-ctx-view-budget=8000", "-compact-history-budget=48000", "-elide-stale-reads", "-ctx-paging=auto"}); err != nil {
		t.Fatalf("parse re-arm: %v", err)
	}
	if *ctxView2 != 8000 || *compact2 != 48000 || !*elideStale2 || *paging2 != gateway.CtxPagingAuto {
		t.Fatalf("explicit re-arm must win: ctxview=%d compact=%d elidestale=%v paging=%q", *ctxView2, *compact2, *elideStale2, *paging2)
	}
}
