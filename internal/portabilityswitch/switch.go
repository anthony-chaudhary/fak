// Package portabilityswitch coordinates context changes with the existing lifecycle
// and process-forest authorities. It owns transaction state, not processes.
//
// Invariant: portability switch evaluations are fail-closed and bounded. Any missing
// authority evidence, stale generation identity, or external unmanaged process
// immediately aborts the transition before mutating execution state.
package portabilityswitch

import (
	"errors"
	"fmt"
	"sort"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/lifecycle"
	"github.com/anthony-chaudhary/fak/internal/processforest"
)

type Capability string

const (
	HotSwitch         Capability = "hot-switch"
	CheckpointRestart Capability = "checkpoint-restart"
	Unsupported       Capability = "unsupported"
)

type Identity struct {
	ID         string `json:"id"`
	Generation uint64 `json:"generation"`
}
type Discovery struct {
	Managed           []Identity `json:"managed"`
	External          []string   `json:"external,omitempty"`
	LifecycleEvidence string     `json:"lifecycle_evidence"`
}
type ACK struct {
	ID         string
	Generation uint64
}
type Checkpoint struct {
	ID            string
	Generation    uint64
	HistoryDigest string
	NextOperation uint64
}

type Adapter interface {
	Capability() Capability
	Apply(transaction, context string) error
}
type Runtime interface {
	Discover(root string) (Discovery, error)
	Quiesce(Identity) (ACK, error)
	Checkpoint(Identity) (Checkpoint, error)
	Resume(Identity, Checkpoint, string) error
	Abort(Identity, Checkpoint) error
	Reconcile(Identity, Checkpoint, string) error
}
type Journal interface {
	Load(string) (Receipt, bool, error)
	Save(Receipt) error
}

type MemberReceipt struct {
	Identity   Identity   `json:"identity"`
	Checkpoint Checkpoint `json:"checkpoint,omitempty"`
	Quiesced   bool       `json:"quiesced"`
	Resumed    bool       `json:"resumed"`
}
type Receipt struct {
	Transaction         string          `json:"transaction"`
	Root                string          `json:"root"`
	Context             string          `json:"context"`
	Capability          Capability      `json:"capability"`
	Status              string          `json:"status"`
	Failure             string          `json:"failure,omitempty"`
	LifecycleEvidence   string          `json:"lifecycle_evidence"`
	PortabilityEvidence string          `json:"portability_evidence"`
	Members             []MemberReceipt `json:"members"`
	External            []string        `json:"external,omitempty"`
	Applied             bool            `json:"applied"`
}
type Request struct{ Transaction, Root, Context string }
type Coordinator struct {
	Adapter Adapter
	Runtime Runtime
	Journal Journal
}

// Switch executes a managed portability switch transaction across registered runtimes and adapters.
//
// Invariant: portability switch evaluations are fail-closed and bounded. Any missing
// authority evidence, stale generation identity, or external unmanaged process
// immediately aborts the transition before mutating execution state.
// Guard: all inputs must be non-empty and all dependencies non-nil; missing authority evidence or
// unmanaged external processes trigger an immediate fail-closed abort prior to adapter mutation.
func (c Coordinator) Switch(q Request) (Receipt, error) {
	if q.Transaction == "" || q.Root == "" || q.Context == "" || c.Adapter == nil || c.Runtime == nil || c.Journal == nil {
		return Receipt{}, errors.New("portability switch: incomplete request")
	}
	if old, ok, err := c.Journal.Load(q.Transaction); err != nil {
		return Receipt{}, err
	} else if ok {
		return c.reconcile(old)
	}
	r := Receipt{Transaction: q.Transaction, Root: q.Root, Context: q.Context, Capability: c.Adapter.Capability(), Status: "preparing"}
	if r.Capability == Unsupported {
		return c.fail(r, "adapter capability unsupported")
	}
	d, err := c.Runtime.Discover(q.Root)
	if err != nil {
		return c.fail(r, "discovery: "+err.Error())
	}
	r.LifecycleEvidence = d.LifecycleEvidence
	r.External = append([]string(nil), d.External...)
	if r.LifecycleEvidence == "" {
		return c.fail(r, "missing lifecycle evidence")
	}
	if len(r.External) > 0 {
		sort.Strings(r.External)
		return c.fail(r, "unmanaged external sessions block switch")
	}
	sort.Slice(d.Managed, func(i, j int) bool { return d.Managed[i].ID < d.Managed[j].ID })
	for _, id := range d.Managed {
		ack, e := c.Runtime.Quiesce(id)
		if e != nil || ack.ID != id.ID || ack.Generation != id.Generation {
			return c.abort(r, "missing ACK or stale identity: "+id.ID)
		}
		mr := MemberReceipt{Identity: id, Quiesced: true}
		r.Members = append(r.Members, mr)
		if r.Capability == CheckpointRestart {
			cp, e := c.Runtime.Checkpoint(id)
			if e != nil || cp.ID != id.ID || cp.Generation != id.Generation || cp.HistoryDigest == "" {
				return c.abort(r, "partial checkpoint: "+id.ID)
			}
			r.Members[len(r.Members)-1].Checkpoint = cp
		}
	}
	if err := c.Journal.Save(r); err != nil {
		return r, err
	}
	if err := c.Adapter.Apply(q.Transaction, q.Context); err != nil {
		return c.abort(r, "apply: "+err.Error())
	}
	r.Applied = true
	r.Status = "resuming"
	r.PortabilityEvidence = "apply:" + q.Transaction
	if err := c.Journal.Save(r); err != nil {
		return r, err
	}
	return c.reconcile(r)
}

func (c Coordinator) reconcile(r Receipt) (Receipt, error) {
	if r.Status == "complete" {
		return r, nil
	}
	if !r.Applied {
		return c.abort(r, "controller restart before apply")
	}
	for i := range r.Members {
		if r.Members[i].Resumed {
			continue
		}
		var err error
		if r.Capability == CheckpointRestart {
			err = c.Runtime.Reconcile(r.Members[i].Identity, r.Members[i].Checkpoint, r.Context)
		} else {
			err = c.Runtime.Resume(r.Members[i].Identity, Checkpoint{}, r.Context)
		}
		if err != nil {
			r.Status = "reconcile-required"
			r.Failure = "resume: " + err.Error()
			_ = c.Journal.Save(r)
			return r, errors.New(r.Failure)
		}
		r.Members[i].Resumed = true
		if err = c.Journal.Save(r); err != nil {
			return r, err
		}
	}
	r.Status = "complete"
	r.Failure = ""
	r.PortabilityEvidence += ";resume-once"
	if err := c.Journal.Save(r); err != nil {
		return r, err
	}
	return r, nil
}
func (c Coordinator) abort(r Receipt, why string) (Receipt, error) {
	for i := range r.Members {
		if r.Members[i].Resumed {
			continue
		}
		if err := c.Runtime.Abort(r.Members[i].Identity, r.Members[i].Checkpoint); err != nil {
			r.Status, r.Failure = "reconcile-required", why+"; abort: "+err.Error()
			_ = c.Journal.Save(r)
			return r, errors.New(r.Failure)
		}
		r.Members[i].Resumed = true
	}
	return c.fail(r, why)
}

func (c Coordinator) fail(r Receipt, why string) (Receipt, error) {
	r.Status = "abstained"
	r.Failure = why
	_ = c.Journal.Save(r)
	return r, errors.New(why)
}

// DiscoveryFromForest composes with the #6432 process controller's snapshot; it
// does not inspect or control OS processes itself.
func DiscoveryFromForest(s processforest.Snapshot, phase lifecycle.Phase, external []string) Discovery {
	d := Discovery{External: append([]string(nil), external...), LifecycleEvidence: fmt.Sprintf("forest:%s;phase:%s", s.ForestID, phase)}
	for _, m := range s.Members {
		if m.State == processforest.StateActive {
			d.Managed = append(d.Managed, Identity{m.MemberID, m.Generation})
		}
	}
	return d
}

type MemoryJournal struct {
	mu       sync.Mutex
	receipts map[string]Receipt
}

func NewMemoryJournal() *MemoryJournal { return &MemoryJournal{receipts: map[string]Receipt{}} }
func (j *MemoryJournal) Load(id string) (Receipt, bool, error) {
	j.mu.Lock()
	defer j.mu.Unlock()
	r, ok := j.receipts[id]
	return r, ok, nil
}
func (j *MemoryJournal) Save(r Receipt) error {
	j.mu.Lock()
	defer j.mu.Unlock()
	j.receipts[r.Transaction] = r
	return nil
}
