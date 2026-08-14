package portabilityswitch

import (
	"errors"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/lifecycle"
	"github.com/anthony-chaudhary/fak/internal/processforest"
)

type fakeAdapter struct {
	capability Capability
	applies    int
	fail       bool
}

func (a *fakeAdapter) Capability() Capability { return a.capability }
func (a *fakeAdapter) Apply(_, _ string) error {
	a.applies++
	if a.fail {
		return errors.New("injected")
	}
	return nil
}

type fakeRuntime struct {
	discovery                            Discovery
	stale, missingACK, missingCheckpoint string
	failResumeOnce                       string
	histories                            map[string][]string
	next                                 map[string]uint64
	resumes                              map[string]int
	work                                 map[string]map[uint64]int
}

func (r *fakeRuntime) Discover(string) (Discovery, error) { return r.discovery, nil }
func (r *fakeRuntime) Quiesce(id Identity) (ACK, error) {
	if id.ID == r.missingACK {
		return ACK{}, errors.New("missing")
	}
	a := ACK{id.ID, id.Generation}
	if id.ID == r.stale {
		a.Generation++
	}
	return a, nil
}
func (r *fakeRuntime) Checkpoint(id Identity) (Checkpoint, error) {
	cp := Checkpoint{id.ID, id.Generation, strings.Join(r.histories[id.ID], "|"), r.next[id.ID]}
	if id.ID == r.missingCheckpoint {
		cp.HistoryDigest = ""
	}
	return cp, nil
}
func (r *fakeRuntime) Abort(id Identity, _ Checkpoint) error {
	r.resumes[id.ID]++
	return nil
}
func (r *fakeRuntime) Resume(id Identity, _ Checkpoint, _ string) error {
	r.resumes[id.ID]++
	return nil
}
func (r *fakeRuntime) Reconcile(id Identity, cp Checkpoint, context string) error {
	if r.failResumeOnce == id.ID {
		r.failResumeOnce = ""
		return errors.New("interrupted")
	}
	// Idempotency key is the checkpoint's next operation. A controller restart
	// can replay reconcile but cannot execute that operation twice.
	if r.work[id.ID] == nil {
		r.work[id.ID] = map[uint64]int{}
	}
	r.work[id.ID][cp.NextOperation]++
	r.histories[id.ID] = append(r.histories[id.ID], context)
	r.next[id.ID] = cp.NextOperation + 1
	r.resumes[id.ID]++
	return nil
}

func fixture(cap Capability) (Coordinator, *fakeAdapter, *fakeRuntime, *MemoryJournal) {
	a := &fakeAdapter{capability: cap}
	rt := &fakeRuntime{
		discovery: Discovery{Managed: []Identity{{"parent", 7}, {"child", 3}}, LifecycleEvidence: "forest:f1@7;phase:active"},
		histories: map[string][]string{"parent": {"p0", "p1"}, "child": {"c0"}}, next: map[string]uint64{"parent": 2, "child": 1},
		resumes: map[string]int{}, work: map[string]map[uint64]int{},
	}
	j := NewMemoryJournal()
	return Coordinator{a, rt, j}, a, rt, j
}

func TestMixedForestRestartResumesExactlyOnceWithHistoryContinuity(t *testing.T) {
	c, a, rt, j := fixture(CheckpointRestart)
	rt.failResumeOnce = "child"
	before := map[string][]string{"parent": append([]string(nil), rt.histories["parent"]...), "child": append([]string(nil), rt.histories["child"]...)}
	r, err := c.Switch(Request{"tx-mixed", "parent", "context-B"})
	if err == nil || r.Status != "reconcile-required" {
		t.Fatalf("interruption=%+v err=%v", r, err)
	}
	// New coordinator instance models controller restart using the durable journal.
	restarted := Coordinator{a, rt, j}
	r, err = restarted.Switch(Request{"tx-mixed", "parent", "context-B"})
	if err != nil || r.Status != "complete" {
		t.Fatalf("restart=%+v err=%v", r, err)
	}
	if a.applies != 1 {
		t.Fatalf("apply count=%d", a.applies)
	}
	for _, id := range []string{"parent", "child"} {
		if rt.resumes[id] != 1 {
			t.Errorf("%s resumes=%d", id, rt.resumes[id])
		}
		if got := rt.work[id][uint64(len(before[id]))]; got != 1 {
			t.Errorf("%s duplicate work=%d", id, got)
		}
		want := append(before[id], "context-B")
		if !reflect.DeepEqual(rt.histories[id], want) {
			t.Errorf("%s history=%v want %v", id, rt.histories[id], want)
		}
	}
	if r.LifecycleEvidence == "" || !strings.Contains(r.PortabilityEvidence, "resume-once") {
		t.Fatalf("unlinked receipt: %+v", r)
	}
}

func TestFailuresAbstainBeforeMutation(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*fakeAdapter, *fakeRuntime)
	}{
		{"unsupported", func(a *fakeAdapter, _ *fakeRuntime) { a.capability = Unsupported }},
		{"unmanaged external", func(_ *fakeAdapter, r *fakeRuntime) { r.discovery.External = []string{"outside-shell"} }},
		{"missing ACK", func(_ *fakeAdapter, r *fakeRuntime) { r.missingACK = "parent" }},
		{"stale identity", func(_ *fakeAdapter, r *fakeRuntime) { r.stale = "parent" }},
		{"partial checkpoint", func(_ *fakeAdapter, r *fakeRuntime) { r.missingCheckpoint = "parent" }},
		{"apply failure", func(a *fakeAdapter, _ *fakeRuntime) { a.fail = true }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c, a, rt, _ := fixture(CheckpointRestart)
			tt.mutate(a, rt)
			r, err := c.Switch(Request{"tx", "parent", "B"})
			if err == nil || r.Status != "abstained" {
				t.Fatalf("receipt=%+v err=%v", r, err)
			}
			if tt.name != "apply failure" && a.applies != 0 {
				t.Fatalf("mutated: applies=%d", a.applies)
			}
			if tt.name == "unsupported" || tt.name == "unmanaged external" {
				if len(rt.resumes) != 0 {
					t.Fatalf("unexpected resume: %v", rt.resumes)
				}
			} else if len(rt.resumes) == 0 {
				t.Fatalf("quiesced work was not restored")
			}
		})
	}
}

func TestHotSwitchSkipsCheckpointAndForestDiscoveryUsesAuthorities(t *testing.T) {
	reg := processforest.NewRegistry()
	if err := reg.Register(processforest.Member{ForestID: "f", MemberID: "root", Generation: 1, State: processforest.StateActive, Observation: processforest.ProcessObservation{PID: 10, StartedAt: time.Unix(1, 0)}, RootAuthority: "controller", AdapterKind: "test"}); err != nil {
		t.Fatal(err)
	}
	s, err := reg.Snapshot("f")
	if err != nil {
		t.Fatal(err)
	}
	d := DiscoveryFromForest(s, lifecycle.Running, nil)
	if len(d.Managed) != 1 || !strings.Contains(d.LifecycleEvidence, "phase:running") {
		t.Fatalf("discovery=%+v", d)
	}
	c, a, rt, _ := fixture(HotSwitch)
	r, err := c.Switch(Request{"hot", "parent", "B"})
	if err != nil || r.Status != "complete" || a.applies != 1 {
		t.Fatalf("receipt=%+v err=%v", r, err)
	}
	if rt.resumes["parent"] != 1 || rt.resumes["child"] != 1 {
		t.Fatalf("resumes=%v", rt.resumes)
	}
}
