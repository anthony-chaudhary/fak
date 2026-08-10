package devcmd

// Shared "planning without live side effects" witness for the fleet/benchmark
// command families that moved to the development artifact under #6022.
//
// The Definition of Done for that migration asks for parity coverage proving the
// fleet/benchmark PLANNING paths still run without live side effects once they
// dispatch from fak-dev. "No live side effect" is made checkable here rather than
// asserted in prose: a planning run must
//
//   1. issue no HTTP round trip through the default transport (no network), and
//   2. leave every byte under its workspace exactly as it found it (no fleet
//      mutation, no ledger append, no run directory, no spawned-worker artifact).
//
// Commands that own an explicit live seam (a spawner, a gh shell-out, a collector)
// additionally inject a fake and assert the call count is zero; this witness covers
// the two effects every family shares.

import (
	"crypto/sha256"
	"encoding/hex"
	"io"
	"io/fs"
	"net/http"
	"os"
	"path/filepath"
	"testing"
)

// refusingTransport fails the test if any code path under witness reaches the
// network through http.DefaultTransport / http.DefaultClient.
type refusingTransport struct {
	t     *testing.T
	calls int
}

func (r *refusingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	r.t.Helper()
	r.calls++
	r.t.Errorf("planning path made a live HTTP request to %s", req.URL)
	return nil, io.EOF
}

// planningWitness records the pre-run state a planning command must not disturb.
type planningWitness struct {
	t         *testing.T
	root      string
	before    map[string]string
	transport *refusingTransport
}

// beginPlanningWitness snapshots root and arms the no-network guard. Callers run
// the planning command and then call assertNoLiveSideEffects.
func beginPlanningWitness(t *testing.T, root string) *planningWitness {
	t.Helper()
	rt := &refusingTransport{t: t}
	saved := http.DefaultTransport
	http.DefaultTransport = rt
	t.Cleanup(func() { http.DefaultTransport = saved })
	return &planningWitness{t: t, root: root, before: snapshotTree(t, root), transport: rt}
}

func (w *planningWitness) assertNoLiveSideEffects() {
	w.t.Helper()
	if w.transport.calls != 0 {
		w.t.Errorf("planning path made %d live HTTP round trips, want 0", w.transport.calls)
	}
	after := snapshotTree(w.t, w.root)
	for path, sum := range after {
		prior, ok := w.before[path]
		switch {
		case !ok:
			w.t.Errorf("planning path created %s under the workspace (live side effect)", path)
		case prior != sum:
			w.t.Errorf("planning path rewrote %s under the workspace (live side effect)", path)
		}
	}
	for path := range w.before {
		if _, ok := after[path]; !ok {
			w.t.Errorf("planning path deleted %s from the workspace (live side effect)", path)
		}
	}
}

// snapshotTree maps every regular file under root to a content digest.
func snapshotTree(t *testing.T, root string) map[string]string {
	t.Helper()
	out := map[string]string{}
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() || !d.Type().IsRegular() {
			return nil
		}
		body, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		sum := sha256.Sum256(body)
		out[filepath.ToSlash(rel)] = hex.EncodeToString(sum[:])
		return nil
	})
	if err != nil {
		t.Fatalf("snapshot %s: %v", root, err)
	}
	return out
}
