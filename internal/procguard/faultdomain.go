package procguard

import (
	"errors"
	"fmt"
	"os"
	"runtime"
	"strings"
	"sync"
	"time"
)

// EnforcementMode states how strongly the operating system enforces an envelope.
type EnforcementMode string

const (
	EnforcementHard        EnforcementMode = "hard"
	EnforcementDegraded    EnforcementMode = "degraded"
	EnforcementObserveOnly EnforcementMode = "observe-only"
)

// ResourceEnvelope is the per-owner resource budget. Zero leaves a limit unset.
// ScratchBytes and OpenFiles are reported but remain advisory until an OS adapter
// can bind them to the complete descendant tree.
type ResourceEnvelope struct {
	MemoryBytes        uint64
	CPUPercent         uint32
	CPUTime            time.Duration
	ProcessCount       uint32
	OpenFiles          uint64
	ScratchBytes       uint64
	CoordinatorReserve ResourceReserve
}

// ResourceReserve records headroom deliberately kept outside the fault domain.
type ResourceReserve struct {
	MemoryBytes  uint64
	CPUPercent   uint32
	ProcessCount uint32
}

// LimitSupport describes one requested resource at the active OS boundary.
type LimitSupport struct {
	Resource string `json:"resource"`
	Enforced bool   `json:"enforced"`
	Detail   string `json:"detail,omitempty"`
}

// FaultDomainReceipt is the durable truth about one owner envelope.
type FaultDomainReceipt struct {
	OwnerID                string          `json:"owner_id"`
	Mode                   EnforcementMode `json:"mode"`
	Primitive              string          `json:"primitive"`
	DescendantsBound       bool            `json:"descendants_bound"`
	Limits                 []LimitSupport  `json:"limits"`
	CoordinatorReserve     ResourceReserve `json:"coordinator_reserve"`
	InvalidatingAssumption string          `json:"invalidating_assumption"`
}

// ResourceUsage is a point-in-time OS usage sample for one owner.
type ResourceUsage struct {
	MemoryBytes uint64        `json:"memory_bytes"`
	CPUTime     time.Duration `json:"cpu_time"`
	Processes   uint64        `json:"processes"`
}

// LimitReceipt is a typed pressure or limit event.
type LimitReceipt struct {
	OwnerID  string `json:"owner_id"`
	Resource string `json:"resource"`
	Usage    uint64 `json:"usage"`
	Limit    uint64 `json:"limit"`
	Exceeded bool   `json:"exceeded"`
}

// FaultDomain owns one OS resource boundary. Close releases only this owner's
// boundary; native kill-on-close behavior may terminate its remaining subtree.
type FaultDomain struct {
	mu       sync.Mutex
	owner    string
	envelope ResourceEnvelope
	receipt  FaultDomainReceipt
	native   nativeFaultDomain
	closed   bool
}

type nativeFaultDomain interface {
	bindCurrent() error
	usage() (ResourceUsage, error)
	close() error
}

// NewFaultDomain creates a uniquely owned envelope without binding any process.
// Call BindCurrent before launching descendants so inheritance is atomic.
func NewFaultDomain(ownerID string, envelope ResourceEnvelope) (*FaultDomain, error) {
	ownerID = strings.TrimSpace(ownerID)
	if ownerID == "" {
		return nil, errors.New("procguard: fault-domain owner is required")
	}
	if envelope.CPUPercent > 100 {
		return nil, errors.New("procguard: CPU percent must be <= 100")
	}
	if envelope.CoordinatorReserve.CPUPercent > 100 {
		return nil, errors.New("procguard: coordinator CPU reserve must be <= 100")
	}
	native, receipt, err := newNativeFaultDomain(ownerID, envelope)
	if err != nil {
		return nil, err
	}
	receipt.OwnerID = ownerID
	receipt.CoordinatorReserve = envelope.CoordinatorReserve
	receipt.InvalidatingAssumption = "the owner remains unprivileged and binds itself before launching descendants"
	return &FaultDomain{owner: ownerID, envelope: envelope, receipt: receipt, native: native}, nil
}

// BindCurrent places the calling fak instance in its boundary. OS inheritance
// then binds all future descendants to the same fault domain.
func (d *FaultDomain) BindCurrent() (FaultDomainReceipt, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed {
		return d.receipt, errors.New("procguard: fault domain is closed")
	}
	if err := d.native.bindCurrent(); err != nil {
		d.receipt.Mode = EnforcementObserveOnly
		d.receipt.DescendantsBound = false
		return d.receipt, fmt.Errorf("procguard: bind %s fault domain: %w", runtime.GOOS, err)
	}
	d.receipt.DescendantsBound = true
	return d.receipt, nil
}

// Receipt returns a copy of the current enforcement contract.
func (d *FaultDomain) Receipt() FaultDomainReceipt {
	d.mu.Lock()
	defer d.mu.Unlock()
	r := d.receipt
	r.Limits = append([]LimitSupport(nil), r.Limits...)
	return r
}

// Usage reads current usage for this owner from the OS boundary.
func (d *FaultDomain) Usage() (ResourceUsage, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed {
		return ResourceUsage{}, errors.New("procguard: fault domain is closed")
	}
	return d.native.usage()
}

// Pressure compares current usage with configured hard limits.
func (d *FaultDomain) Pressure() ([]LimitReceipt, error) {
	u, err := d.Usage()
	if err != nil {
		return nil, err
	}
	checks := []struct {
		name         string
		usage, limit uint64
	}{{"memory", u.MemoryBytes, d.envelope.MemoryBytes}, {"cpu_time", uint64(u.CPUTime), uint64(d.envelope.CPUTime)}, {"processes", u.Processes, uint64(d.envelope.ProcessCount)}}
	out := make([]LimitReceipt, 0, len(checks))
	for _, c := range checks {
		if c.limit > 0 {
			out = append(out, LimitReceipt{OwnerID: d.owner, Resource: c.name, Usage: c.usage, Limit: c.limit, Exceeded: c.usage >= c.limit})
		}
	}
	return out, nil
}

// Close releases only this owner's OS object.
func (d *FaultDomain) Close() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed {
		return nil
	}
	d.closed = true
	return d.native.close()
}

func requestedSupport(e ResourceEnvelope, enforced map[string]string) []LimitSupport {
	requested := []struct {
		name string
		on   bool
	}{{"memory", e.MemoryBytes > 0}, {"cpu_share", e.CPUPercent > 0}, {"cpu_time", e.CPUTime > 0}, {"processes", e.ProcessCount > 0}, {"open_files", e.OpenFiles > 0}, {"scratch", e.ScratchBytes > 0}}
	var out []LimitSupport
	for _, r := range requested {
		if !r.on {
			continue
		}
		detail, ok := enforced[r.name]
		if !ok {
			detail = "no whole-tree OS primitive in this adapter"
		}
		out = append(out, LimitSupport{Resource: r.name, Enforced: ok, Detail: detail})
	}
	return out
}

func modeFor(limits []LimitSupport) EnforcementMode {
	if len(limits) == 0 {
		return EnforcementObserveOnly
	}
	for _, l := range limits {
		if !l.Enforced {
			return EnforcementDegraded
		}
	}
	return EnforcementHard
}

func sanitizeFaultDomainID(s string) string {
	var b strings.Builder
	for _, r := range s {
		if r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '-' || r == '_' {
			b.WriteRune(r)
		} else {
			b.WriteByte('_')
		}
	}
	if b.Len() == 0 {
		return fmt.Sprintf("pid-%d", os.Getpid())
	}
	return b.String()
}
