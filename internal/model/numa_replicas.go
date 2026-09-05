package model

import (
	"fmt"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type numaModelState struct {
	mu                  sync.Mutex
	numaReplicasLabel   string
	numaReplicasEnabled bool
	numaSchedule        compute.DecodeNUMASchedule
	numaTopology        []compute.NUMANodeTopology
	numaPool            *compute.NUMADecodePool
}

func (m *Model) ensureNUMAState() *numaModelState {
	if m.numa == nil {
		m.numa = &numaModelState{}
	}
	return m.numa
}

// ApplyNUMAWeightReplicas evaluates the NUMA replica request (from flag or FAK_NUMA_REPLICAS),
// configures per-NUMA-node replicas for all resident Q4_K and k-quant weights,
// and initializes a barrier-free per-node decode schedule.
func (m *Model) ApplyNUMAWeightReplicas(req string) string {
	if m == nil {
		return "numa_replicas=nil_model"
	}
	st := m.ensureNUMAState()
	st.mu.Lock()
	defer st.mu.Unlock()

	cfg := compute.ResolveNUMAReplicaConfig(req)
	if !cfg.Enabled || len(cfg.Topology) < 2 {
		_ = m.freeNUMAReplicasLocked(st)
		st.numaReplicasEnabled = false
		st.numaReplicasLabel = fmt.Sprintf("numa_replicas=disabled(reason=%s)", cfg.Reason)
		return st.numaReplicasLabel
	}

	workers := q4kDecodeWorkers()
	sched := compute.ScheduleDecodeNUMA(cfg.Topology, workers)
	if !sched.Eligible {
		_ = m.freeNUMAReplicasLocked(st)
		st.numaReplicasEnabled = false
		st.numaReplicasLabel = fmt.Sprintf("numa_replicas=refused(reason=%s)", sched.Reason)
		return st.numaReplicasLabel
	}

	pool, err := compute.NewNUMADecodePool(sched)
	if err != nil {
		_ = m.freeNUMAReplicasLocked(st)
		st.numaReplicasEnabled = false
		st.numaReplicasLabel = fmt.Sprintf("numa_replicas=pool_failed(%v)", err)
		return st.numaReplicasLabel
	}

	if st.numaPool != nil {
		_ = st.numaPool.Close()
	}
	st.numaPool = pool
	st.numaSchedule = sched
	st.numaTopology = cfg.Topology

	replicatedCount := 0
	replicatedBytes := int64(0)

	// Replicate all resident Q4_K tensors
	for _, qt := range m.q4kw {
		if qt == nil || len(qt.raw) == 0 {
			continue
		}
		if qt.replicas != nil {
			_ = qt.replicas.Free()
			qt.replicas = nil
		}
		set, err := compute.BuildNUMAReplicasForTopology(qt.raw, cfg.Topology)
		if err != nil {
			_ = m.freeNUMAReplicasLocked(st)
			st.numaReplicasEnabled = false
			st.numaReplicasLabel = fmt.Sprintf("numa_replicas=alloc_failed(%v)", err)
			return st.numaReplicasLabel
		}
		qt.replicas = set
		qt.numaPool = pool
		replicatedCount++
		replicatedBytes += int64(len(qt.raw)) * int64(len(cfg.Topology))
	}

	if m.q4khead != nil && len(m.q4khead.raw) > 0 {
		if m.q4khead.replicas == nil {
			set, err := compute.BuildNUMAReplicasForTopology(m.q4khead.raw, cfg.Topology)
			if err == nil {
				m.q4khead.replicas = set
				m.q4khead.numaPool = pool
				replicatedCount++
				replicatedBytes += int64(len(m.q4khead.raw)) * int64(len(cfg.Topology))
			}
		} else {
			m.q4khead.numaPool = pool
		}
	}

	// Also replicate resident k-quant tensors
	for _, qt := range m.kqw {
		if qt == nil || len(qt.raw) == 0 {
			continue
		}
		if qt.replicas != nil {
			_ = qt.replicas.Free()
			qt.replicas = nil
		}
		set, err := compute.BuildNUMAReplicasForTopology(qt.raw, cfg.Topology)
		if err == nil {
			qt.replicas = set
			qt.numaPool = pool
			replicatedCount++
			replicatedBytes += int64(len(qt.raw)) * int64(len(cfg.Topology))
		}
	}

	st.numaReplicasEnabled = true
	st.numaReplicasLabel = fmt.Sprintf("numa_replicas=applied(nodes=%d,workers=%d,tensors=%d,bytes=%d)",
		len(cfg.Topology), sched.Workers, replicatedCount, replicatedBytes)
	return st.numaReplicasLabel
}

// NUMAReplicasLabel returns the cached status of NUMA replicas on this model.
func (m *Model) NUMAReplicasLabel() string {
	if m == nil || m.numa == nil || m.numa.numaReplicasLabel == "" {
		return "numa_replicas=unrun"
	}
	st := m.ensureNUMAState()
	st.mu.Lock()
	defer st.mu.Unlock()
	if st.numaReplicasLabel == "" {
		return "numa_replicas=unrun"
	}
	return st.numaReplicasLabel
}

// NUMAReplicasEnabled reports whether NUMA weight replication and the barrier-free schedule
// are active on this model.
func (m *Model) NUMAReplicasEnabled() bool {
	if m == nil || m.numa == nil {
		return false
	}
	st := m.ensureNUMAState()
	st.mu.Lock()
	defer st.mu.Unlock()
	return st.numaReplicasEnabled
}

// NUMADecodePool returns the active NUMA decode worker pool, or nil when disabled.
func (m *Model) NUMADecodePool() *compute.NUMADecodePool {
	if m == nil || m.numa == nil {
		return nil
	}
	st := m.ensureNUMAState()
	st.mu.Lock()
	defer st.mu.Unlock()
	return st.numaPool
}

// NUMASchedule returns the active NUMA decode schedule.
func (m *Model) NUMASchedule() compute.DecodeNUMASchedule {
	if m == nil || m.numa == nil {
		return compute.DecodeNUMASchedule{}
	}
	st := m.ensureNUMAState()
	st.mu.Lock()
	defer st.mu.Unlock()
	return st.numaSchedule
}

// NUMATopology returns the active NUMA node topology.
func (m *Model) NUMATopology() []compute.NUMANodeTopology {
	if m == nil || m.numa == nil {
		return nil
	}
	st := m.ensureNUMAState()
	st.mu.Lock()
	defer st.mu.Unlock()
	return append([]compute.NUMANodeTopology(nil), st.numaTopology...)
}

// FreeNUMAReplicas releases all allocated weight replicas and shuts down the NUMA decode pool.
func (m *Model) FreeNUMAReplicas() error {
	if m == nil || m.numa == nil {
		return nil
	}
	st := m.ensureNUMAState()
	st.mu.Lock()
	defer st.mu.Unlock()
	return m.freeNUMAReplicasLocked(st)
}

func (m *Model) freeNUMAReplicasLocked(st *numaModelState) error {
	var firstErr error
	if st.numaPool != nil {
		if err := st.numaPool.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
		st.numaPool = nil
	}
	for _, qt := range m.q4kw {
		if qt != nil && qt.replicas != nil {
			if err := qt.replicas.Free(); err != nil && firstErr == nil {
				firstErr = err
			}
			qt.replicas = nil
			qt.numaPool = nil
		}
	}
	if m.q4khead != nil && m.q4khead.replicas != nil {
		if err := m.q4khead.replicas.Free(); err != nil && firstErr == nil {
			firstErr = err
		}
		m.q4khead.replicas = nil
		m.q4khead.numaPool = nil
	}
	for _, qt := range m.kqw {
		if qt != nil && qt.replicas != nil {
			if err := qt.replicas.Free(); err != nil && firstErr == nil {
				firstErr = err
			}
			qt.replicas = nil
			qt.numaPool = nil
		}
	}
	st.numaReplicasEnabled = false
	return firstErr
}
