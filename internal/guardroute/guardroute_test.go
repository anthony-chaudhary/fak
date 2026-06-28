package guardroute

import (
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/guardrsi"
)

func TestDecideOverBuckets(t *testing.T) {
	cases := []struct {
		name      string
		fold      guardrsi.Fold
		bucket    guardrsi.Bucket
		threshold int
		wantRoute bool
		wantSev   string
		wantIssue bool
		wantCause string
	}{
		{
			name:      "blank reason on deny -> P1 + issue",
			fold:      guardrsi.Fold{TotalRows: 5, BlankReasonOnDeny: 2},
			bucket:    guardrsi.Bucket{Bucket: "blank_reason_on_deny", Count: 2, Lever: "require a reason"},
			wantRoute: true, wantSev: SevP1, wantIssue: true,
			wantCause: "guard-journal:blank_reason_on_deny",
		},
		{
			name:      "unknown verdict -> P1 + issue",
			fold:      guardrsi.Fold{TotalRows: 4, UnknownVerdict: 1},
			bucket:    guardrsi.Bucket{Bucket: "unknown_verdict", Count: 1, Lever: "closed set"},
			wantRoute: true, wantSev: SevP1, wantIssue: true,
			wantCause: "guard-journal:unknown_verdict",
		},
		{
			name:      "recurring denial at threshold -> P2 queue-only",
			fold:      guardrsi.Fold{TotalRows: 10},
			bucket:    guardrsi.Bucket{Bucket: "reason:POLICY_BLOCK", Count: 3, Lever: "floor"},
			threshold: 3,
			wantRoute: true, wantSev: SevP2, wantIssue: false,
			wantCause: "guard-journal:reason:POLICY_BLOCK",
		},
		{
			name:      "denial below threshold -> no route",
			fold:      guardrsi.Fold{TotalRows: 10},
			bucket:    guardrsi.Bucket{Bucket: "reason:POLICY_BLOCK", Count: 2, Lever: "floor"},
			threshold: 3,
			wantRoute: false,
		},
		{
			name:      "clean fold (bucket none) -> no route",
			fold:      guardrsi.Fold{TotalRows: 7},
			bucket:    guardrsi.Bucket{Bucket: "none", Lever: "nothing to retire"},
			wantRoute: false,
		},
		{
			name:      "empty journal -> no route, self-diagnosing",
			fold:      guardrsi.Fold{TotalRows: 0},
			bucket:    guardrsi.Bucket{Bucket: "none"},
			wantRoute: false,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			d := Decide(tc.fold, tc.bucket, tc.threshold)
			if d.Route != tc.wantRoute {
				t.Fatalf("Route = %v, want %v (reason: %s)", d.Route, tc.wantRoute, d.Reason)
			}
			if d.Reason == "" {
				t.Fatalf("Reason must never be empty (it is the self-diagnosing why/why-not)")
			}
			if !tc.wantRoute {
				return
			}
			if d.Severity != tc.wantSev {
				t.Fatalf("Severity = %q, want %q", d.Severity, tc.wantSev)
			}
			if d.FileIssue != tc.wantIssue {
				t.Fatalf("FileIssue = %v, want %v", d.FileIssue, tc.wantIssue)
			}
			if d.CauseKey != tc.wantCause {
				t.Fatalf("CauseKey = %q, want %q", d.CauseKey, tc.wantCause)
			}
			if d.Pattern == "" || d.Item == "" {
				t.Fatalf("routed decision must carry a Pattern and Item: %+v", d)
			}
		})
	}
}

func TestDecideIsPure(t *testing.T) {
	fold := guardrsi.Fold{TotalRows: 5, BlankReasonOnDeny: 2}
	bucket := guardrsi.Bucket{Bucket: "blank_reason_on_deny", Count: 2, Lever: "x"}
	a := Decide(fold, bucket, 0)
	b := Decide(fold, bucket, 0)
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("Decide is not pure:\n a=%+v\n b=%+v", a, b)
	}
}

func TestDecideDefaultThreshold(t *testing.T) {
	fold := guardrsi.Fold{TotalRows: 10}
	// count 3 with threshold 0 should use DefaultReasonThreshold (3) -> routes.
	d := Decide(fold, guardrsi.Bucket{Bucket: "reason:X", Count: DefaultReasonThreshold}, 0)
	if !d.Route {
		t.Fatalf("count == DefaultReasonThreshold should route under the default; got %+v", d)
	}
	// one below should not.
	d = Decide(fold, guardrsi.Bucket{Bucket: "reason:X", Count: DefaultReasonThreshold - 1}, 0)
	if d.Route {
		t.Fatalf("count below default threshold should not route; got %+v", d)
	}
}

func TestToActionItemStableKey(t *testing.T) {
	d := Decide(guardrsi.Fold{TotalRows: 3, BlankReasonOnDeny: 1},
		guardrsi.Bucket{Bucket: "blank_reason_on_deny", Count: 1, Lever: "x"}, 0)
	item := ToActionItem(d, "/evidence.jsonl")
	if item.Key != "guard-rsi-route/guard-journal:blank_reason_on_deny" {
		t.Fatalf("unstable issue key: %q", item.Key)
	}
	if item.EvidencePath != "/evidence.jsonl" {
		t.Fatalf("evidence path not carried: %q", item.EvidencePath)
	}
	// The same decision must map to the same key (so a re-run updates in place).
	if ToActionItem(d, "/other.jsonl").Key != item.Key {
		t.Fatalf("issue key must be evidence-independent / stable")
	}
}

func TestRouteArgvShape(t *testing.T) {
	d := Decide(guardrsi.Fold{TotalRows: 3, UnknownVerdict: 1},
		guardrsi.Bucket{Bucket: "unknown_verdict", Count: 1, Lever: "x"}, 0)
	argv := RouteArgv(d, "sess-7")
	if argv[0] != "route_finding" {
		t.Fatalf("argv must start with the route_finding subcommand: %v", argv)
	}
	joined := strings.Join(argv, " ")
	for _, want := range []string{
		"--cause-key guard-journal:unknown_verdict",
		"--sev P1",
		"--source sess-7",
		"--key sess-7:guard-journal:unknown_verdict",
		"--owning-plan GUARD-RSI",
	} {
		if !strings.Contains(joined, want) {
			t.Fatalf("argv missing %q\n got: %s", want, joined)
		}
	}
	// empty source falls back to a stable default.
	if got := RouteArgv(d, ""); !strings.Contains(strings.Join(got, " "), "--source guard-verdict-rsi") {
		t.Fatalf("empty source should default to guard-verdict-rsi: %v", got)
	}
}

func TestFoldEnvelopeAlwaysOK(t *testing.T) {
	// A routed finding is the pass working -> OK=true, verdict ACTION.
	routed := Decide(guardrsi.Fold{TotalRows: 3, BlankReasonOnDeny: 1},
		guardrsi.Bucket{Bucket: "blank_reason_on_deny", Count: 1, Lever: "x"}, 0)
	e := Fold(routed, map[string]any{"action": "routed", "n": 1}, nil)
	if !e.OK || e.Verdict != "ACTION" || e.Finding != "guard_route_routed" {
		t.Fatalf("routed envelope = %+v", e)
	}
	// A clear review -> OK=true, verdict OK.
	clear := Decide(guardrsi.Fold{TotalRows: 3}, guardrsi.Bucket{Bucket: "none"}, 0)
	e = Fold(clear, nil, nil)
	if !e.OK || e.Verdict != "OK" || e.Finding != "guard_route_clear" {
		t.Fatalf("clear envelope = %+v", e)
	}
	if e.NextAction == "" {
		t.Fatalf("envelope must always carry a next_action")
	}
}
