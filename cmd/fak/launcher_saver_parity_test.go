package main

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

// TestLaunchSurfacesPreserveDefaultSavers is the on-by-default-for-headless/ultracode lock: every
// launch surface that assembles its OWN `fak guard … --` argv (instead of reaching guard through
// the interactive front door the token-defaults scorecard reads) must inherit guard's default-on
// token-saver stack — i.e. inject NO flag that disables compaction, oversized-result elision, the
// O(1) planned view, or vDSO. If a future edit splices a saver-disabling override into any of
// these builders, this fails while the front-door scorecard stays green.
func TestLaunchSurfacesPreserveDefaultSavers(t *testing.T) {
	headlessRaw, ok := dispatchtick.GuardedLaunchCommand(
		[]string{"claude", "-p", "resolve the issue"}, "fak", "core", "claude", "/repo", "")
	if !ok {
		t.Fatal("GuardedLaunchCommand refused to front a claude worker with guard")
	}

	cases := []struct {
		name string
		argv []string
	}{
		{
			name: "accounts launch — default (guard + ultracode + pinned model)",
			argv: buildLaunchArgv("fak", launchOpts{
				command: "claude", useGuard: true, skipPermissions: true,
				ultracode: true, model: defaultLaunchModel,
			}),
		},
		{
			name: "accounts launch — ultracode + managed-cache on",
			argv: buildLaunchArgv("fak", launchOpts{
				command: "claude", useGuard: true, ultracode: true,
				guardCacheArgs: guardCachePostureArgs(guardManagedCacheOn, ""),
			}),
		},
		{
			name: "accounts launch — codex seat (managed-cache off posture)",
			argv: buildLaunchArgv("fak", launchOpts{
				command: "codex", useGuard: true, skipPermissions: true,
				guardCacheArgs: guardCachePostureArgs(guardManagedCacheOff, ""),
			}),
		},
		{
			name: "headless dispatch worker",
			argv: headlessRaw,
		},
		{
			name: "headless dispatch worker + managed-cache posture spliced",
			argv: spliceGuardPostureArgs(headlessRaw, guardCachePostureArgs(guardManagedCacheOn, "ANTHROPIC_API_KEY")),
		},
		{
			name: "codex launcher — quiet (headless-style) + managed-cache on",
			argv: buildCodexLaunchArgv("fak", codexLaunchOptions{
				splitMode: "auto", splitWhere: "bottom", quiet: true,
				skipPermissions: true, managedCache: guardManagedCacheOn,
			}),
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := guardArgvDisabledSavers(tc.argv); got != nil {
				t.Fatalf("launch surface disables on-by-default savers %v\n  argv: %v", got, tc.argv)
			}
		})
	}
}

// TestGuardArgvDisabledSaversDetects proves the invariant actually fires — a lock whose detector
// never triggers is no lock. Each row is a saver-disabling override a launcher edit might
// introduce, in every flag form guard accepts, plus the boundaries the detector must NOT trip on.
func TestGuardArgvDisabledSaversDetects(t *testing.T) {
	cases := []struct {
		name string
		argv []string
		want []string
	}{
		{"compaction off (space form)", []string{"fak", "guard", "--compact-history-budget", "0", "--", "claude"}, []string{"compacthistory"}},
		{"compaction off (eq form)", []string{"fak", "guard", "--compact-history-budget=0", "--", "claude"}, []string{"compacthistory"}},
		{"elision off (negative)", []string{"fak", "guard", "--elide-result-bytes", "-1", "--", "claude"}, []string{"elideresult"}},
		{"ctxview off (eq form)", []string{"fak", "guard", "--ctx-view-budget=0", "--", "claude"}, []string{"ctxview"}},
		{"vdso off (eq false)", []string{"fak", "guard", "--vdso=false", "--", "claude"}, []string{"vdso"}},
		{"vdso off (eq 0)", []string{"fak", "guard", "--vdso=0", "--", "claude"}, []string{"vdso"}},
		{"vdso off (negated flag)", []string{"fak", "guard", "--no-vdso", "--", "claude"}, []string{"vdso"}},
		{"two savers off, sorted", []string{"fak", "guard", "--vdso=off", "--compact-history-budget=0", "--", "claude"}, []string{"compacthistory", "vdso"}},
		{"stale elision off (eq false)", []string{"fak", "guard", "--elide-stale-reads=false", "--", "claude"}, []string{"elidestale"}},
		{"stale elision off (eq 0)", []string{"fak", "guard", "--elide-stale-reads=0", "--", "claude"}, []string{"elidestale"}},
		{"three savers off, sorted", []string{"fak", "guard", "--elide-stale-reads=off", "--vdso=false", "--ctx-view-budget=0", "--", "claude"}, []string{"ctxview", "elidestale", "vdso"}},

		{"positive budget keeps saver on", []string{"fak", "guard", "--ctx-view-budget", "8000", "--", "claude"}, nil},
		{"bare --elide-stale-reads keeps saver on", []string{"fak", "guard", "--elide-stale-reads", "--", "claude"}, nil},
		{"--elide-stale-reads=true keeps saver on", []string{"fak", "guard", "--elide-stale-reads=true", "--", "claude"}, nil},
		{"bare --vdso keeps saver on", []string{"fak", "guard", "--vdso", "--", "claude"}, nil},
		{"--vdso=true keeps saver on", []string{"fak", "guard", "--vdso=true", "--", "claude"}, nil},
		{"disable after -- is the agent's, not guard's", []string{"fak", "guard", "--", "claude", "--ctx-view-budget", "0"}, nil},
		{"no guard segment (unguarded)", []string{"claude", "-p", "go"}, nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := guardArgvDisabledSavers(tc.argv); !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("guardArgvDisabledSavers(%v) = %v, want %v", tc.argv, got, tc.want)
			}
		})
	}
}
