package dispatchtick

import (
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/hostplacement"
)

// spillPrompt is a realistic worker prompt: it carries spaces, punctuation and
// an apostrophe, so a remote hand-off that forgets to quote tears it apart.
const spillPrompt = "resolve #3599: it's a two-host wave"

// tickWorkerArgv rebuilds the argv the LIVE dispatch tick hands its spawner:
// BuildWorkerCommand renders the backend command and GuardedLaunchCommand
// fronts it with `fak guard` — the exact composition cmd/fak/dispatch_tick.go's
// dispatchTickLiveSpawn performs before spawning. Placing THIS argv (rather
// than a hand-written fixture) is what makes the witness a tick-path witness
// instead of another pure-function table.
func tickWorkerArgv(t *testing.T, backend string, guard bool) []string {
	t.Helper()
	base, err := BuildWorkerCommand(backend, spillPrompt, WorkerLaunch{Model: "a-model", Fallback: "a-fallback", Effort: "xhigh"})
	if err != nil {
		t.Fatalf("BuildWorkerCommand(%q): %v", backend, err)
	}
	if !guard {
		return base
	}
	guarded, ok := GuardedLaunchCommand(base, "fak", "docs", backend, "/repo", "http://127.0.0.1:18080")
	if !ok {
		t.Fatalf("GuardedLaunchCommand refused to front the %s worker", backend)
	}
	return guarded
}

// TestPlaceWorkerLaunchSpillsThroughTickPath is the launch-seam DoD: it runs the
// REAL tick worker argv through the placement consult and asserts what would
// actually be spawned. The pure-function table (TestResolveHostSpill) proves the
// DECISION; this proves the decision reaches the command.
func TestPlaceWorkerLaunchSpillsThroughTickPath(t *testing.T) {
	now := spillBase
	argv := tickWorkerArgv(t, "claude", true)

	staleB := hostplacement.NewRegistry()
	staleB.Observe(hostplacement.Heartbeat{Hostname: "host-a", LiveHeadroom: 1, Saturation: 0.95, TS: now})
	staleB.Observe(hostplacement.Heartbeat{Hostname: "host-b", LiveHeadroom: 9, Saturation: 0.05, TS: now.Add(-spillTTL - time.Second)})

	cases := []struct {
		name       string
		spill      HostSpill
		wantRemote bool
		wantHost   string
		wantReason string
	}{
		{
			// Acceptance 1, at the launch seam: host-a (this box) is over the
			// saturation threshold, so the next unit's argv is routed to host-b.
			name:       "primary_saturated_spills_next_unit_to_second_host",
			spill:      HostSpill{LocalHost: "host-a", HostsRaw: "host-a,host-b", Registry: fakeSpillRegistry(0.95, 0.30), Threshold: 0.90, TTL: spillTTL},
			wantRemote: true,
			wantHost:   "host-b",
			wantReason: PlacementRemote,
		},
		{
			// Acceptance 2, at the launch seam: host-b's heartbeat predates the
			// TTL, so it is unavailable and the argv is NEVER routed to it —
			// even though host-a is full and host-b looks idle.
			name:       "stale_second_host_is_never_launched_on",
			spill:      HostSpill{LocalHost: "host-a", HostsRaw: "host-a,host-b", Registry: staleB, Threshold: 0.90, TTL: spillTTL},
			wantReason: hostplacement.ReasonNoneEligible,
		},
		{
			// A placed host that IS the local box is not a spill: the argv must
			// stay verbatim, not be re-entered through the remote executor.
			name:       "primary_with_headroom_launches_locally_unchanged",
			spill:      HostSpill{LocalHost: "host-a", HostsRaw: "host-a,host-b", Registry: fakeSpillRegistry(0.10, 0.30), Threshold: 0.90, TTL: spillTTL},
			wantHost:   "host-a",
			wantReason: PlacementLocal,
		},
		{
			// Acceptance 3, at the launch seam: no FAK_DISPATCH_HOSTS, so the
			// registry is irrelevant and the argv is untouched.
			name:       "single_host_default_launches_locally_unchanged",
			spill:      HostSpill{LocalHost: "host-a", Registry: fakeSpillRegistry(0.10, 0.10), Threshold: 0.90, TTL: spillTTL},
			wantReason: hostplacement.ReasonNoHosts,
		},
		{
			// Without a local identity the seam cannot tell a spill from a stay,
			// so it refuses to rewrite. It still REPORTS the placed host, so the
			// tick can log why it declined.
			name:       "unknown_local_identity_refuses_to_spill",
			spill:      HostSpill{HostsRaw: "host-a,host-b", Registry: fakeSpillRegistry(0.95, 0.05), Threshold: 0.90, TTL: spillTTL},
			wantHost:   "host-b",
			wantReason: PlacementLocalHostUnknown,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.spill.PlaceLaunch(argv, now)
			if got.Remote != tc.wantRemote {
				t.Fatalf("Remote = %v, want %v (placement %+v)", got.Remote, tc.wantRemote, got)
			}
			if got.Host != tc.wantHost {
				t.Fatalf("Host = %q, want %q (placement %+v)", got.Host, tc.wantHost, got)
			}
			if got.Reason != tc.wantReason {
				t.Fatalf("Reason = %q, want %q", got.Reason, tc.wantReason)
			}
			if again := tc.spill.PlaceLaunch(argv, now); !reflect.DeepEqual(again, got) {
				t.Fatalf("non-deterministic placement: %+v then %+v", got, again)
			}

			if !tc.wantRemote {
				if !reflect.DeepEqual(got.Command, argv) {
					t.Fatalf("local launch rewrote the argv:\n got %q\nwant %q", got.Command, argv)
				}
				// The local argv must be a COPY, never an alias: a caller that
				// keeps mutating its own slice cannot reach into the placement.
				got.Command[0] = "mutated"
				if argv[0] == "mutated" {
					t.Fatal("PlaceLaunch aliased the caller's argv instead of copying it")
				}
			} else {
				if len(got.Command) != 3 {
					t.Fatalf("remote argv = %q, want [%s <host> <command>]", got.Command, DefaultRemoteLaunchProgram)
				}
				if got.Command[0] != DefaultRemoteLaunchProgram || got.Command[1] != tc.wantHost {
					t.Fatalf("remote argv targets %q via %q, want host %q via %q", got.Command[1], got.Command[0], tc.wantHost, DefaultRemoteLaunchProgram)
				}
				// The whole tick argv — guard front, model flags and prompt —
				// survives inside the single remote word.
				for _, want := range argv {
					if !strings.Contains(got.Command[2], shellQuoteArg(want)) {
						t.Fatalf("remote command %q dropped argv element %q", got.Command[2], want)
					}
				}
				if !strings.Contains(got.Command[2], "fak guard") {
					t.Fatalf("remote command %q lost the guard front", got.Command[2])
				}
			}
		})
	}
}

// TestPlaceWorkerLaunchSingleHostDefaultIsByteIdentical is acceptance item 3 —
// the regression that matters when a new decision is wired into a live path.
// With no FAK_DISPATCH_HOSTS the placed argv must be byte-identical to the argv
// the tick built, for EVERY backend and both guarded and unguarded, and even
// though the registry is stuffed with perfectly idle, perfectly fresh hosts the
// seam would happily spill to if the env gate ever stopped holding. Threshold
// and TTL are left zero here, so the package defaults are on the hook too.
func TestPlaceWorkerLaunchSingleHostDefaultIsByteIdentical(t *testing.T) {
	now := spillBase
	idle := fakeSpillRegistry(0, 0)

	for _, backend := range []string{"claude", "opencode", "codex"} {
		for _, guard := range []bool{false, true} {
			name := backend
			if guard {
				name += "_guarded"
			}
			t.Run(name, func(t *testing.T) {
				argv := tickWorkerArgv(t, backend, guard)
				before := strings.Join(argv, "\x00")

				got := HostSpill{LocalHost: "host-a", Registry: idle}.PlaceLaunch(argv, now)

				if got.Remote || got.Host != "" {
					t.Fatalf("single-host default placed remotely: %+v", got)
				}
				if got.Reason != hostplacement.ReasonNoHosts {
					t.Fatalf("Reason = %q, want %q", got.Reason, hostplacement.ReasonNoHosts)
				}
				if after := strings.Join(got.Command, "\x00"); after != before {
					t.Fatalf("single-host default argv changed:\n got %q\nwant %q", got.Command, argv)
				}
			})
		}
	}
}

// TestRemoteLaunchCommandSurvivesRemoteShellSplit pins the remote hand-off
// shape: exactly three words, the worker command collapsed into one
// shell-quoted word so the far-side shell cannot re-split the prompt.
func TestRemoteLaunchCommandSurvivesRemoteShellSplit(t *testing.T) {
	argv := []string{"fak", "guard", "--provider", "anthropic", "--", "claude", "-p", spillPrompt}
	got := remoteLaunchCommand(DefaultRemoteLaunchProgram, "host-b", argv)

	want := []string{
		DefaultRemoteLaunchProgram,
		"host-b",
		`fak guard --provider anthropic -- claude -p 'resolve #3599: it'\''s a two-host wave'`,
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("remoteLaunchCommand =\n %q\nwant\n %q", got, want)
	}
	if remoteLaunchCommand(DefaultRemoteLaunchProgram, "host-b", nil) != nil {
		t.Fatal("an empty argv must render no remote command")
	}
}

// TestPlaceWorkerLaunchEmptyCommand proves the seam never invents a remote
// launch out of nothing: with a spill decision in hand but no argv to place,
// it reports no_command and rewrites nothing.
func TestPlaceWorkerLaunchEmptyCommand(t *testing.T) {
	spill := HostSpill{LocalHost: "host-a", HostsRaw: "host-a,host-b", Registry: fakeSpillRegistry(0.95, 0.05), Threshold: 0.90, TTL: spillTTL}
	got := spill.PlaceLaunch(nil, spillBase)
	if got.Remote || len(got.Command) != 0 || got.Reason != PlacementNoCommand {
		t.Fatalf("PlaceLaunch(nil) = %+v, want no_command with no argv", got)
	}
}
