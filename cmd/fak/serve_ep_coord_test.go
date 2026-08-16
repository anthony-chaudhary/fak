package main

import (
	"strings"
	"testing"
)

// TestResolveEPDecodeRoleMatrix pins the whole role/refusal matrix of the coordinated EP
// decode wiring (#4835). Every arm here is a decision the serve makes ONCE at boot on a
// GPU box with no debugger attached, where the alternative to a refusal is a deadlock
// (mirror-vs-coordinated frame interleave) or a wrong answer (a reduce that sums a partial
// computed from a different context). Keeping it pure — the caller passes the env — is what
// makes the matrix checkable without a process group.
func TestResolveEPDecodeRoleMatrix(t *testing.T) {
	sharded0 := epRankConfig{ranks: 8, rank: 0, coordAddr: "127.0.0.1:9000", sharded: true}
	sharded3 := epRankConfig{ranks: 8, rank: 3, coordAddr: "127.0.0.1:9000", sharded: true}
	plain := epRankConfig{ranks: 1}

	for _, tc := range []struct {
		name        string
		cfg         epRankConfig
		coord       string
		fanout      string
		radix       string
		want        epDecodeRole
		wantErrPart string
	}{
		// Default-off: not opted in, nothing changes anywhere, including on a sharded serve
		// that is still running today's HTTP request mirror.
		{name: "unset/plain", cfg: plain, want: epDecodeRequestMirror},
		{name: "unset/sharded-rank0", cfg: sharded0, want: epDecodeRequestMirror},
		{name: "unset/sharded-follower", cfg: sharded3, want: epDecodeRequestMirror},
		{name: "off/sharded", cfg: sharded0, coord: "0", want: epDecodeRequestMirror},
		{name: "mirror-still-allowed-when-off", cfg: sharded0, coord: "false", fanout: "http://a:1,http://b:1", want: epDecodeRequestMirror},

		// Opted in and correctly configured: the asymmetric roles.
		{name: "on/rank0-coordinates", cfg: sharded0, coord: "1", radix: "off", want: epDecodeCoordinator},
		{name: "on/rank3-follows", cfg: sharded3, coord: "true", radix: "off", want: epDecodeFollower},
		{name: "on/vocabulary-YES", cfg: sharded3, coord: " YES ", radix: "off", want: epDecodeFollower},
		{name: "on/vocabulary-on", cfg: sharded0, coord: "on", radix: " OFF ", want: epDecodeCoordinator},

		// Refusals. Each returns the safe role alongside the error so a caller that logged
		// instead of exiting still cannot end up in the coordinated topology.
		{name: "refuse/not-sharded", cfg: plain, coord: "1", radix: "off", want: epDecodeRequestMirror, wantErrPart: "not a sharded expert-parallel rank"},
		{name: "refuse/mirror-also-configured", cfg: sharded0, coord: "1", fanout: "http://follower-1:8080", radix: "off", want: epDecodeRequestMirror, wantErrPart: "mutually exclusive topologies"},
		{name: "refuse/mirror-also-configured-follower", cfg: sharded3, coord: "1", fanout: " http://follower-1:8080 ", radix: "off", want: epDecodeRequestMirror, wantErrPart: "FAK_EP_FANOUT_ADDRS"},
		{name: "refuse/radix-default-on", cfg: sharded0, coord: "1", radix: "", want: epDecodeRequestMirror, wantErrPart: "FAK_INKERNEL_RADIX=off"},
		{name: "refuse/radix-explicitly-on", cfg: sharded3, coord: "1", radix: "on", want: epDecodeRequestMirror, wantErrPart: "FAK_INKERNEL_RADIX=off"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := resolveEPDecodeRole(tc.cfg, tc.coord, tc.fanout, tc.radix)
			if got != tc.want {
				t.Fatalf("resolveEPDecodeRole(coord=%q fanout=%q radix=%q) role = %v, want %v", tc.coord, tc.fanout, tc.radix, got, tc.want)
			}
			switch {
			case tc.wantErrPart == "" && err != nil:
				t.Fatalf("resolveEPDecodeRole(coord=%q fanout=%q radix=%q) = %v, want no error", tc.coord, tc.fanout, tc.radix, err)
			case tc.wantErrPart != "" && err == nil:
				t.Fatalf("resolveEPDecodeRole(coord=%q fanout=%q radix=%q) succeeded, want refusal naming %q", tc.coord, tc.fanout, tc.radix, tc.wantErrPart)
			case tc.wantErrPart != "" && !strings.Contains(err.Error(), tc.wantErrPart):
				t.Fatalf("resolveEPDecodeRole error = %v, want it to name %q", err, tc.wantErrPart)
			}
		})
	}
}

// TestEPDecodeRoleExactlyOneRankCoordinates is the property the whole topology rests on: a
// coordinated world has exactly one rank that binds a listener, tokenizes, and samples, and
// every other rank contributes only local expert work. If two ranks ever resolved to
// coordinator they would both broadcast frames into the same DistComm; if zero did, the
// followers would park forever. The HTTP mirror this replaces has the opposite property —
// all N ranks run the whole request, which is what the 8-GPU witness measured at
// 0.0406 tok/s against a ~0.2 tok/s scalar baseline.
func TestEPDecodeRoleExactlyOneRankCoordinates(t *testing.T) {
	const ranks = 8
	coordinators, followers := 0, 0
	for rank := range ranks {
		cfg := epRankConfig{ranks: ranks, rank: rank, coordAddr: "127.0.0.1:9000", sharded: true}
		role, err := resolveEPDecodeRole(cfg, "1", "", "off")
		if err != nil {
			t.Fatalf("rank %d: resolveEPDecodeRole: %v", rank, err)
		}
		switch role {
		case epDecodeCoordinator:
			coordinators++
			if rank != 0 {
				t.Fatalf("rank %d resolved as coordinator; only rank 0 may broadcast frames", rank)
			}
		case epDecodeFollower:
			followers++
		default:
			t.Fatalf("rank %d resolved as %v under FAK_EP_COORDINATED_DECODE=1", rank, role)
		}
	}
	if coordinators != 1 || followers != ranks-1 {
		t.Fatalf("coordinated world of %d resolved to %d coordinator(s) + %d follower(s), want 1 + %d", ranks, coordinators, followers, ranks-1)
	}
}

// TestEPDecodeRoleDefaultIsInert pins that an un-opted-in serve is byte-identical to today:
// applyEPDecodeRole must not touch the model, install a coordinator, or park the process,
// which is what makes landing this wiring reversible on trunk while the end-to-end tok/s
// claim is still GPU-gated. A nil model is the sharp end of that — the coordinator and
// follower arms both refuse a nil model loudly, so reaching either one here would fail.
func TestEPDecodeRoleDefaultIsInert(t *testing.T) {
	rt := &serveRuntime{ep: epRankConfig{ranks: 8, rank: 0, sharded: true}, epRole: epDecodeRequestMirror}
	applyEPDecodeRole(rt, nil)
	if rt.epCoord != nil {
		t.Fatalf("request-mirror role installed a decode coordinator (%v)", rt.epCoord)
	}
}

// TestEPDecodeCoordinatorRefusesNilModel pins the two fail-closed guards the boot path leans
// on to turn a misconfigured coordinated serve into an exit code instead of a hang.
func TestEPDecodeCoordinatorRefusesNilModel(t *testing.T) {
	if _, err := installEPDecodeCoordinator(nil, nil); err == nil {
		t.Fatal("installEPDecodeCoordinator(nil group, nil model) succeeded; want a refusal")
	}
	if err := runEPFollowerRank(nil, nil, nil); err == nil {
		t.Fatal("runEPFollowerRank(nil group, nil model) succeeded; want a refusal")
	}
}

// TestCoordinatedEPReachableFromTheSanctionedLauncher pins that the sanctioned launcher can actually
// SELECT the topology this wiring adds. Without this arm the coordinated path is dead code on
// the sanctioned hardware: tools/glm52_fak_native_serve.sh routes EP_FRONTEND_FANOUT!=1 to
// smoke_ready_ep, which curls EVERY rank port — and a coordinated follower binds no listener, so
// seven of eight curls fail and the launcher exits SMOKE_FAIL on a healthy serve. The three
// forced env settings are the ones `fak serve` refuses at boot if they disagree
// (resolveEPDecodeRole above), and learning that after a 466 GB stage is expensive.
func TestCoordinatedEPReachableFromTheSanctionedLauncher(t *testing.T) {
	launcher := readRepoTextForClaudeGLMGCP(t, repoRootFromTest(t), "tools", "glm52_fak_native_serve.sh")
	for _, want := range []string{
		"FAK_EP_COORDINATED_DECODE=1",
		"FAK_INKERNEL_RADIX=off",
		"EP_FRONTEND_FANOUT=0",
	} {
		requireContainsForClaudeGLMGCP(t, launcher, want)
	}
	witness := readRepoTextForClaudeGLMGCP(t, repoRootFromTest(t), "tools", "glm52_ep_witness.sh")
	for _, want := range []string{"EP_PREFLIGHT", "LOAD_READY", "completion_tokens"} {
		requireContainsForClaudeGLMGCP(t, witness, want)
	}
}

// TestEPDecodeRoleStrings pins the operator-facing names — they appear in the boot lines that
// tell an operator which topology a rank actually came up in, which is the only signal that
// distinguishes a coordinated run from a mirror run in a captured witness log.
func TestEPDecodeRoleStrings(t *testing.T) {
	for role, want := range map[epDecodeRole]string{
		epDecodeRequestMirror: "request-mirror",
		epDecodeCoordinator:   "coordinator",
		epDecodeFollower:      "follower",
	} {
		if got := role.String(); got != want {
			t.Fatalf("epDecodeRole(%d).String() = %q, want %q", int(role), got, want)
		}
	}
}
