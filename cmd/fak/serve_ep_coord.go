package main

// FENCED, not finished: this file does not compile against committed symbols yet — it uses
// rt.epRole and rt.epCoord, and serveRuntime declares neither. `go build` compiles every
// non-test .go file in the package whether or not it is committed, so while this sat
// untagged it reddened `go build ./cmd/fak` for every session sharing this checkout (~39h).
// AGENTS.md's remedy for exactly this case is the tag above, which keeps the work on disk
// and the default build green. Build it with `go build -tags wip_serve_ep_coord ./cmd/fak`,
// and DROP THIS TAG once epRole/epCoord land on serveRuntime. Tag added by a peer session,
// not by this file's author; nothing below was changed.

// serve_ep_coord.go — the `fak serve` wiring for the COORDINATED expert-parallel decode
// (#4835), the half internal/model/ep_decode_coord.go named as "not covered here".
//
// The protocol landed already: model.EPDecodeCoordinator broadcasts the forward rank 0 is
// about to run, model.RunEPFollower replays it, and the seam is Session.Prefill/Step so
// the serve's OWN decode loop drives the group. What was missing is the wiring that picks
// a rank's role at boot: without it every rank still bound an HTTP listener and the only
// way to reach the collectives was internal/gateway/http_epfanout.go's request mirror —
// which runs the WHOLE request (tokenize, prefill, decode, sample) N times and returns
// only rank 0's body. That mirror is what the sanctioned 8-GPU witness measured at 0.0406 tok/s,
// slower than the ~0.2 tok/s scalar pure-fak baseline.
//
// Under the coordinated topology the roles are asymmetric and the asymmetry is the point:
//
//   - rank 0    binds the listener, owns tokenization/sampling/the token history, and
//               announces every forward to the group;
//   - ranks>0   bind NOTHING. They park in model.RunEPFollower and contribute only their
//               local expert compute + collectives, so a request is tokenized once and
//               sampled once no matter how wide the world is.
//
// Selection is an explicit opt-in (FAK_EP_COORDINATED_DECODE). Default-off keeps every
// existing sharded serve byte-identical to today's mirror topology, which is what makes
// this reversible on trunk while the end-to-end tok/s claim stays GPU-gated.

import (
	"fmt"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
)

// epDecodeRole is this process's place in the EP decode topology — the closed set of
// answers resolveEPDecodeRole returns.
type epDecodeRole int

const (
	// epDecodeRequestMirror is today's default: no coordination, every rank serves HTTP and
	// the front rank mirrors the inbound request to the followers (http_epfanout.go).
	epDecodeRequestMirror epDecodeRole = iota
	// epDecodeCoordinator is rank 0 of a coordinated serve: it serves HTTP and drives the
	// followers from its own decode loop through model.EPDecodeCoordinator.
	epDecodeCoordinator
	// epDecodeFollower is rank>0 of a coordinated serve: it never binds a listener and never
	// samples; it parks in model.RunEPFollower until rank 0 broadcasts SHUTDOWN.
	epDecodeFollower
)

func (r epDecodeRole) String() string {
	switch r {
	case epDecodeCoordinator:
		return "coordinator"
	case epDecodeFollower:
		return "follower"
	default:
		return "request-mirror"
	}
}

// epEnvTruthy is the shared on/off reading for this file's env gates, matching
// epRequireDevicePG's vocabulary so an operator sets every EP switch the same way.
func epEnvTruthy(raw string) bool {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

// resolveEPDecodeRole resolves this process's EP decode role from the already-resolved rank
// config plus the three env inputs the coordinated path is sensitive to, failing closed on
// every misconfig that would otherwise surface as a deadlock or a wrong answer on a GPU box
// nobody can attach a debugger to. It is pure — the caller supplies the env — so the whole
// refusal matrix is checkable without a process group.
//
// The refusals, and why each one is fatal rather than a warning:
//
//   - Coordination asked for on a serve that is not sharded. FAK_EP_COORDINATED_DECODE
//     needs a real process group to broadcast on; without one it is a silent no-op that
//     would leave an operator believing they measured the coordinated path.
//   - Coordination asked for WITH the HTTP request mirror still configured. The two
//     topologies are mutually exclusive by construction: the mirror hands a follower its own
//     copy of the request, so the follower would run rank 0's frames AND its own independent
//     decode, interleaving two frame streams on one DistComm. That is the desync the
//     protocol's mirror-position guard exists to catch, and there is no reason to reach it.
//   - Coordination asked for with KV-prefix reuse still on. This is the one named in
//     RunEPFollower's contract: a rank-0 session restored from a radix prefix hit prefills
//     only the divergent SUFFIX, at a sequence position the follower's fresh mirror never
//     computed. The follower fails closed there, so the serve refuses here instead — one
//     boot-time refusal naming the fix beats a request-time collapse of the whole group.
//     Refusing (rather than quietly forcing the tree off) also keeps the serve honest about
//     what the operator gives up: the sanctioned warm witness measured 0.214 tok/s on an
//     exact-prefix repeat versus 0.0917 tok/s on a distinct prompt, so that cache is a real
//     and measurable effect and its loss must not be silent.
func resolveEPDecodeRole(cfg epRankConfig, coordEnv, fanoutAddrs, radixEnv string) (epDecodeRole, error) {
	if !epEnvTruthy(coordEnv) {
		// Not requested: today's topology, unchanged, including on a sharded serve.
		return epDecodeRequestMirror, nil
	}
	if !cfg.sharded {
		return epDecodeRequestMirror, fmt.Errorf("FAK_EP_COORDINATED_DECODE is set but this serve is not a sharded expert-parallel rank (needs --expert-parallel N>1 AND FAK_EP_COORD_ADDR); there is no process group to coordinate over")
	}
	if strings.TrimSpace(fanoutAddrs) != "" {
		return epDecodeRequestMirror, fmt.Errorf("FAK_EP_COORDINATED_DECODE and FAK_EP_FANOUT_ADDRS are mutually exclusive topologies: the coordinated path drives followers from rank 0's decode loop, while the HTTP mirror re-runs the whole request on every rank. Unset FAK_EP_FANOUT_ADDRS on every rank (#4835)")
	}
	if !strings.EqualFold(strings.TrimSpace(radixEnv), "off") {
		return epDecodeRequestMirror, fmt.Errorf("FAK_EP_COORDINATED_DECODE requires FAK_INKERNEL_RADIX=off: a rank-0 session resumed from a KV-prefix hit prefills only the divergent suffix, at a position the follower's mirror never computed, and model.RunEPFollower fails that request closed. Set FAK_INKERNEL_RADIX=off on every rank (#4835)")
	}
	if cfg.rank == 0 {
		return epDecodeCoordinator, nil
	}
	return epDecodeFollower, nil
}

// epFollowerSessionFactory mints a follower rank's mirror sessions. RunEPFollower's contract
// is that the mirror MUST be the same KIND of session rank 0 decodes with, because it has to
// run the same forward — so this mirrors the choice the in-kernel planner makes for rank 0
// (internal/agent/inkernel_decode.go: a device serve decodes on a backend session, a host
// serve on the plain one). Getting this wrong would not fail loudly; it would run a different
// forward and reduce a mismatched partial.
func epFollowerSessionFactory(m *fakmodel.Model, be compute.Backend) func() *fakmodel.Session {
	if be != nil {
		return func() *fakmodel.Session { return m.NewBackendSession(be) }
	}
	return m.NewSession
}

// runEPFollowerRank parks a follower rank in the coordinated decode loop. It BLOCKS until
// rank 0 broadcasts SHUTDOWN (nil) or the group desyncs/fails (error), and it is the reason a
// coordinated follower never reaches the listener: this rank's whole job for the lifetime of
// the serve is to replay rank 0's forwards.
func runEPFollowerRank(group *fakmodel.DistComm, m *fakmodel.Model, be compute.Backend) error {
	if m == nil {
		return fmt.Errorf("expert-parallel follower rank has no loaded model to mirror rank 0's forward with")
	}
	return fakmodel.RunEPFollower(group, epFollowerSessionFactory(m, be))
}

// installEPDecodeCoordinator gives rank 0 the driver that announces every Prefill/Step its
// sessions run. It is installed on the Model BEFORE the gateway builds its in-kernel planner,
// so the planner's ordinary decode loop — not a bespoke driver — is what drives the group.
func installEPDecodeCoordinator(group *fakmodel.DistComm, m *fakmodel.Model) (*fakmodel.EPDecodeCoordinator, error) {
	if m == nil {
		return nil, fmt.Errorf("expert-parallel rank 0 has no loaded model to install the decode coordinator on")
	}
	coord, err := fakmodel.NewEPDecodeCoordinator(group)
	if err != nil {
		return nil, err
	}
	m.SetEPDecodeCoordinator(coord)
	return coord, nil
}

// configureEPDecode resolves the topology after the model and process group are ready,
// then installs rank 0's coordinator or parks a follower before any HTTP listener binds.
func (rt *serveRuntime) configureEPDecode() {
	role, err := resolveEPDecodeRole(rt.ep, os.Getenv("FAK_EP_COORDINATED_DECODE"), os.Getenv("FAK_EP_FANOUT_ADDRS"), os.Getenv("FAK_INKERNEL_RADIX"))
	if err != nil {
		fmt.Fprintln(os.Stderr, "fak serve:", err)
		os.Exit(2)
	}
	rt.epRole = role
	applyEPDecodeRole(rt, rt.inKernelModel)
}

// applyEPDecodeRole performs this rank's resolved role once the group is formed and the
// weights are resident. On a follower it never returns: the rank parks in the decode loop and
// then exits the process, because a coordinated follower has no listener to bind, no tokenizer
// to load, and no session plane to restore.
func applyEPDecodeRole(rt *serveRuntime, m *fakmodel.Model) {
	switch rt.epRole {
	case epDecodeCoordinator:
		coord, err := installEPDecodeCoordinator(rt.epGroup, m)
		if err != nil {
			fmt.Fprintf(os.Stderr, "fak serve: install the coordinated EP decode driver on rank %d/%d: %v\n", rt.ep.rank, rt.ep.ranks, err)
			os.Exit(2)
		}
		rt.epCoord = coord
		rt.addStartupMessage(newServeStartupMessage("expert-parallel", "decode-topology", "info",
			fmt.Sprintf("rank 0/%d owns tokenization and sampling; ranks 1-%d contribute local expert work only; HTTP request mirror off", rt.ep.ranks, rt.ep.ranks-1)))
	case epDecodeFollower:
		err := runEPFollowerRank(rt.epGroup, m, rt.chatBackend)
		// os.Exit skips cmdServe's deferred close, so release the group here on both arms.
		rt.closeEPGroup()
		if err != nil {
			fmt.Fprintf(os.Stderr, "fak serve: expert-parallel follower rank %d/%d left the coordinated decode: %v\n", rt.ep.rank, rt.ep.ranks, err)
			os.Exit(1)
		}
		fmt.Printf("fak: expert-parallel rank %d/%d released by rank 0's SHUTDOWN — exiting cleanly\n", rt.ep.rank, rt.ep.ranks)
		os.Exit(0)
	}
}
