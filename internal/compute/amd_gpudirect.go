// Package compute implements hardware abstraction, tensor computation, memory slab management,
// and zero-copy device interconnect acceleration for the fak agent kernel.
package compute

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"time"
)

// SmallBARThresholdBytes defines the 256 MiB boundary of legacy small PCIe BAR0/BAR1 windows.
// When ReBAR (Resizable BAR) / Large BAR is disabled, modern GPUs with 16-192 GiB VRAM are constrained
// to this 256 MiB aperture, degrading P2P and RDMA throughput by 80-90% due to dynamic page remapping.
const SmallBARThresholdBytes = 256 * 1024 * 1024 // 256 MiB

// MaxSGELengthBytes is the maximum byte span expressible in a single standard 32-bit InfiniBand SGE.
const MaxSGELengthBytes = math.MaxUint32

// AMDFabricType identifies the physical interconnect fabric connecting AMD GPUs and peer devices.
type AMDFabricType string

const (
	// FabricXGMI represents AMD Infinity Fabric point-to-point coherent links (MI200, MI300X).
	// Provides hardware cache-coherency, low latency (~180-250ns), and aggregate bandwidth up to 896 GB/s.
	// Bypasses the PCIe bus and root complex entirely.
	FabricXGMI AMDFabricType = "InfinityFabric_xGMI"

	// FabricPCIeSwitch represents peer-to-peer DMA over a local PCIe Gen4/Gen5 switch (e.g. Microchip/Broadcom PEX).
	// Bypasses the CPU Root Complex with bandwidth up to ~64 GB/s (Gen5 x16) and latency ~400-800ns.
	FabricPCIeSwitch AMDFabricType = "PCIe_Switch_P2P"

	// FabricPCIeHostBridge represents peer transfers that must traverse the host CPU Root Complex / UPI / Infinity Fabric CPU socket link.
	// Subject to inter-socket NUMA bottlenecks and potential PCIe ACS request-redirect drops.
	FabricPCIeHostBridge AMDFabricType = "PCIe_Host_Bridge"

	// FabricDirectStorage represents direct peer-to-peer DMA between an NVMe controller and GPU BAR1 VRAM (BaM / SPDK).
	FabricDirectStorage AMDFabricType = "NVMe_PCIe_P2PDMA"

	// FabricNone indicates no direct peer connection exists.
	FabricNone AMDFabricType = "None"
)

// AMDDeviceNode describes an AMD GPU device node, its PCIe topology, and memory aperture configuration.
type AMDDeviceNode struct {
	NodeID         int        `json:"node_id"`
	GPUID          int        `json:"gpu_id"`
	DeviceName     string     `json:"device_name"`
	Architecture   string     `json:"architecture"` // e.g. "gfx942" (MI300X), "gfx1151" (Strix Halo), "gfx1100" (RX 7900)
	PCIeBDF        string     `json:"pcie_bdf"`     // Bus/Device/Function e.g. "0000:41:00.0"
	NUMANode       int        `json:"numa_node"`
	TotalVRAMBytes uint64     `json:"total_vram_bytes"`
	BAR1SizeBytes  uint64     `json:"bar1_size_bytes"`
	IsLargeBAR     bool       `json:"is_large_bar"`
	ACSEnabled     bool       `json:"acs_enabled"`      // PCIe Access Control Services on upstream bridge
	ACSRedirect    bool       `json:"acs_redirect"`     // True if ACS Request Redirect (RR) is active (breaks PCIe P2P)
	KeepVRAMMapped bool       `json:"keep_vram_mapped"` // amdgpu keep_vram_mapped module parameter
	DMABUFCapable  bool       `json:"dmabuf_capable"`   // Kernel DMA-BUF export/import capability
	Peers          []PeerLink `json:"peers"`
}

// PeerLink describes peer-to-peer connectivity between two AMD device nodes.
type PeerLink struct {
	TargetNodeID     int           `json:"target_node_id"`
	Fabric           AMDFabricType `json:"fabric"`
	BandwidthGBps    float64       `json:"bandwidth_gbps"`
	LatencyNanos     uint32        `json:"latency_nanos"`
	DirectP2PCapable bool          `json:"direct_p2p_capable"`
	Coherent         bool          `json:"coherent"`
	Warning          string        `json:"warning,omitempty"`
}

// DMABUFHandle represents an exported Linux DMA-BUF file descriptor backed by AMD GPU VRAM.
// Implements the user-space contract for DRM PRIME and KFD DMA-BUF export (AMDKFD_IOC_EXPORT_DMABUF).
type DMABUFHandle struct {
	FD           int     `json:"fd"`
	Size         uint64  `json:"size"`
	Offset       uint64  `json:"offset"`
	AlignedSize  uint64  `json:"aligned_size"`
	VRAMAddress  uintptr `json:"vram_address"`
	NodeID       int     `json:"node_id"`
	ExportOffset uint64  `json:"export_offset"`
	Closed       bool    `json:"closed"`
}

// RDMARegisteredRegion represents an AMD GPU VRAM buffer registered directly with InfiniBand/RoCE
// verbs subsystem using Linux DMA-BUF (ibv_reg_dmabuf_mr / ROCm-RDMA kfd_peerdirect).
// All data transfers targeting this region achieve host-bypass zero-copy line rate.
type RDMARegisteredRegion struct {
	RKey        uint32                 `json:"rkey"`
	LKey        uint32                 `json:"lkey"`
	IOVA        uint64                 `json:"iova"`
	Length      uint64                 `json:"length"`
	DMABUFFD    int                    `json:"dmabuf_fd"`
	NodeID      int                    `json:"node_id"`
	SGEs        []ScatterGatherElement `json:"sges"`
	StagingCopy int                    `json:"staging_copy_count"` // Invariant: must be 0
	Active      bool                   `json:"active"`
}

// StagingCopyCount returns the count of intermediate CPU bounce buffer staging copies.
// Under AMD GPU Direct, this invariant is always 0.
func (r *RDMARegisteredRegion) StagingCopyCount() int {
	return r.StagingCopy
}

// NVMeOpcode defines standard NVMe NVM command opcodes.
const (
	// NVMeOpcodeWrite represents NVMe NVM command Write (0x01).
	NVMeOpcodeWrite uint8 = 0x01
	// NVMeOpcodeRead represents NVMe NVM command Read (0x02).
	NVMeOpcodeRead uint8 = 0x02
)

// NVMeP2PCommand represents a direct NVMe storage command (Read/Write) targeting AMD GPU VRAM.
// Inspired by BaM (Big Accelerator Memory) and SPDK ROCm plugins, the NVMe submission queue entry (SQE)
// contains physical PRP/SGL entries directly pointing to GPU BAR1 VRAM.
type NVMeP2PCommand struct {
	CommandID      uint16  `json:"command_id"`
	Opcode         uint8   `json:"opcode"` // 0x02 = Read, 0x01 = Write
	NamespaceID    uint32  `json:"namespace_id"`
	StartingLBA    uint64  `json:"starting_lba"`
	BlockCount     uint16  `json:"block_count"`
	TargetVRAMAddr uintptr `json:"target_vram_addr"`
	ByteLength     uint64  `json:"byte_length"`
	Completed      bool    `json:"completed"`
	Status         uint16  `json:"status"` // NVMe status code (0 = success)
	DurationNanos  int64   `json:"duration_nanos"`
}

// StagingCopyCount returns the number of intermediate host DRAM staging copies for NVMe direct P2P DMA.
// Under BaM / SPDK, this invariant is always 0.
func (cmd *NVMeP2PCommand) StagingCopyCount() int {
	return 0
}

// HSAMemorySignal models an aligned 64-bit atomic completion signal in GPU coherent memory (hsa_signal_t).
// Enables sub-microsecond wait-on-memory synchronization (ISA s_waitcnt) without CPU thread wakeups.
// The value field is at offset 0 to guarantee 64-bit alignment on all architectures.
type HSAMemorySignal struct {
	value    atomic.Int64
	waiters  atomic.Int64
	SignalID string  `json:"signal_id"`
	Address  uintptr `json:"address"`
}

// NewHSAMemorySignal allocates a new HSA coherent memory signal with an initial value.
func NewHSAMemorySignal(id string, initialValue int64, addr uintptr) *HSAMemorySignal {
	s := &HSAMemorySignal{
		SignalID: id,
		Address:  addr,
	}
	s.value.Store(initialValue)
	return s
}

// LoadRelaxed reads the current value of the signal.
func (s *HSAMemorySignal) LoadRelaxed() int64 {
	return s.value.Load()
}

// StoreRelease atomically sets the signal value with release semantics, waking any polling device or host waiter.
func (s *HSAMemorySignal) StoreRelease(v int64) {
	s.value.Store(v)
}

// WaitRelaxed polls the signal value until it matches target or timeout expires.
// Simulates GPU wavefront s_waitcnt memory polling with sub-microsecond spin.
func (s *HSAMemorySignal) WaitRelaxed(target int64, timeout time.Duration) (bool, error) {
	deadline := time.Now().Add(timeout)
	s.waiters.Add(1)
	defer s.waiters.Add(-1)

	// Short active spin simulating GPU ISA s_waitcnt memory polling
	for i := 0; i < 5000; i++ {
		if s.value.Load() == target {
			return true, nil
		}
	}

	for time.Now().Before(deadline) {
		if s.value.Load() == target {
			return true, nil
		}
		time.Sleep(10 * time.Microsecond)
	}

	return false, fmt.Errorf("hsa_signal %s timed out waiting for target %d (current=%d)", s.SignalID, target, s.value.Load())
}

// HSADoorbell models an aligned 64-bit hardware dispatch doorbell (AQL packet submission).
// Host CPU or peer GPU wavefronts ring the doorbell to notify the AMD GPU Command Processor (CP)
// of newly submitted AQL packets with sub-microsecond latency.
type HSADoorbell struct {
	value   atomic.Uint64
	ID      string  `json:"doorbell_id"`
	Address uintptr `json:"address"`
	QueueID uint32  `json:"queue_id"`
}

// NewHSADoorbell allocates a new HSA dispatch doorbell.
func NewHSADoorbell(id string, addr uintptr, queueID uint32) *HSADoorbell {
	return &HSADoorbell{
		ID:      id,
		Address: addr,
		QueueID: queueID,
	}
}

// Ring atomically stores the packet write index with release memory semantics, ringing the doorbell.
func (d *HSADoorbell) Ring(packetIndex uint64) {
	d.value.Store(packetIndex)
}

// ReadRelaxed reads the latest packet index written to the doorbell.
func (d *HSADoorbell) ReadRelaxed() uint64 {
	return d.value.Load()
}

// WaitPacket polls the doorbell until the packet index reaches target or timeout expires.
// Simulates CP doorbell polling with sub-microsecond spin before backing off.
func (d *HSADoorbell) WaitPacket(target uint64, timeout time.Duration) (bool, error) {
	deadline := time.Now().Add(timeout)
	for i := 0; i < 5000; i++ {
		if d.value.Load() >= target {
			return true, nil
		}
	}
	for time.Now().Before(deadline) {
		if d.value.Load() >= target {
			return true, nil
		}
		time.Sleep(10 * time.Microsecond)
	}
	return false, fmt.Errorf("hsa_doorbell %s timed out waiting for packet %d (current=%d)", d.ID, target, d.value.Load())
}

// AMDGPUDirectConfig defines configuration and topology hints for AMDGPUDirectHAL.
type AMDGPUDirectConfig struct {
	EnableLargeBARCheck    bool   `json:"enable_large_bar_check"`
	EnforceACSZeroRedirect bool   `json:"enforce_acs_zero_redirect"`
	PreferXGMI             bool   `json:"prefer_xgmi"`
	DefaultPageSize        uint64 `json:"default_page_size"`
}

// AMDGPUDirectHAL coordinates AMD GPU Direct memory operations: topology discovery,
// DMA-BUF export/import, RDMA verbs registration, NVMe P2P storage streaming, and HSA signal synchronization.
type AMDGPUDirectHAL struct {
	cfg        AMDGPUDirectConfig
	mu         sync.RWMutex
	nodes      map[int]*AMDDeviceNode
	dmabufs    map[int]*DMABUFHandle
	rdmaMRs    map[uint32]*RDMARegisteredRegion
	signals    map[string]*HSAMemorySignal
	doorbells  map[string]*HSADoorbell
	queuePairs map[uint32]*RDMAQueuePair
	transfers  int64
	bytesMoved uint64
	nextFD     int
	nextKey    uint32
	nextQPN    uint32
}

// NewAMDGPUDirectHAL initializes an AMD GPU Direct HAL coordinator.
func NewAMDGPUDirectHAL(cfg AMDGPUDirectConfig) *AMDGPUDirectHAL {
	if cfg.DefaultPageSize == 0 {
		cfg.DefaultPageSize = 4096
	}
	return &AMDGPUDirectHAL{
		cfg:        cfg,
		nodes:      make(map[int]*AMDDeviceNode),
		dmabufs:    make(map[int]*DMABUFHandle),
		rdmaMRs:    make(map[uint32]*RDMARegisteredRegion),
		signals:    make(map[string]*HSAMemorySignal),
		doorbells:  make(map[string]*HSADoorbell),
		queuePairs: make(map[uint32]*RDMAQueuePair),
		nextFD:     100,
		nextKey:    0x2000,
		nextQPN:    0x1000,
	}
}

// RegisterNode registers an AMD GPU device node with the coordinator.
func (e *AMDGPUDirectHAL) RegisterNode(node AMDDeviceNode) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if node.TotalVRAMBytes == 0 {
		return errors.New("amddirect: device node TotalVRAMBytes must be greater than 0")
	}

	// Sizing BAR1 aperture: check for ReBAR / Large BAR
	if node.BAR1SizeBytes == 0 {
		node.BAR1SizeBytes = SmallBARThresholdBytes
	}
	node.IsLargeBAR = node.BAR1SizeBytes >= node.TotalVRAMBytes

	cp := node
	cp.Peers = make([]PeerLink, len(node.Peers))
	copy(cp.Peers, node.Peers)
	e.nodes[node.NodeID] = &cp
	return nil
}

// RouteEntry represents a resolved peer route between two device nodes in a TopologyMatrix.
type RouteEntry struct {
	DirectP2PCapable bool          `json:"direct_p2p_capable"`
	Fabric           AMDFabricType `json:"fabric"`
	BandwidthGBps    float64       `json:"bandwidth_gbps"`
	LatencyNanos     uint32        `json:"latency_nanos"`
	Reason           string        `json:"reason,omitempty"`
}

// TopologyMatrix represents the complete N x N peer-to-peer route and bandwidth matrix across all discovered nodes.
type TopologyMatrix struct {
	NodeIDs []int                      `json:"node_ids"`
	Routes  map[int]map[int]RouteEntry `json:"routes"`
}

// JSON encodes the TopologyMatrix as indented JSON bytes.
func (tm TopologyMatrix) JSON() ([]byte, error) {
	return json.MarshalIndent(tm, "", "  ")
}

// DiscoverTopology returns the active AMD device topology, connectivity matrix, and hardware posture.
func (e *AMDGPUDirectHAL) DiscoverTopology() []AMDDeviceNode {
	e.mu.RLock()
	defer e.mu.RUnlock()

	res := make([]AMDDeviceNode, 0, len(e.nodes))
	for _, n := range e.nodes {
		cp := *n
		cp.Peers = make([]PeerLink, len(n.Peers))
		copy(cp.Peers, n.Peers)
		res = append(res, cp)
	}
	return res
}

// TopologyMatrix computes and returns the complete N x N peer route matrix across all registered nodes.
func (e *AMDGPUDirectHAL) TopologyMatrix() TopologyMatrix {
	e.mu.RLock()
	defer e.mu.RUnlock()

	nodeIDs := make([]int, 0, len(e.nodes))
	for id := range e.nodes {
		nodeIDs = append(nodeIDs, id)
	}

	routes := make(map[int]map[int]RouteEntry, len(nodeIDs))
	for _, srcID := range nodeIDs {
		routes[srcID] = make(map[int]RouteEntry, len(nodeIDs))
		for _, dstID := range nodeIDs {
			ok, fabric, reason := e.validateP2PRouteLocked(srcID, dstID)
			var bw float64
			var lat uint32
			if ok {
				if srcID == dstID {
					fabric = FabricXGMI
					bw = 896.0
					lat = 50
				} else {
					src := e.nodes[srcID]
					for _, p := range src.Peers {
						if p.TargetNodeID == dstID {
							bw = p.BandwidthGBps
							lat = p.LatencyNanos
							break
						}
					}
					if bw == 0 {
						bw = 32.0 // fallback PCIe Gen4
						lat = 800
					}
				}
			}
			routes[srcID][dstID] = RouteEntry{
				DirectP2PCapable: ok,
				Fabric:           fabric,
				BandwidthGBps:    bw,
				LatencyNanos:     lat,
				Reason:           reason,
			}
		}
	}

	return TopologyMatrix{
		NodeIDs: nodeIDs,
		Routes:  routes,
	}
}

// ValidateP2PRoute verifies whether two AMD devices can perform direct peer-to-peer DMA without CPU bounce.
func (e *AMDGPUDirectHAL) ValidateP2PRoute(srcNodeID, dstNodeID int) (bool, AMDFabricType, string) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.validateP2PRouteLocked(srcNodeID, dstNodeID)
}

func (e *AMDGPUDirectHAL) validateP2PRouteLocked(srcNodeID, dstNodeID int) (bool, AMDFabricType, string) {
	src, okSrc := e.nodes[srcNodeID]
	dst, okDst := e.nodes[dstNodeID]
	if !okSrc || !okDst {
		return false, FabricNone, fmt.Sprintf("invalid nodes: src=%d exists=%v, dst=%d exists=%v", srcNodeID, okSrc, dstNodeID, okDst)
	}

	if srcNodeID == dstNodeID {
		return true, FabricXGMI, "intra-device local transfer"
	}

	// First inspect direct peer links.
	// Critical architectural fact: Direct xGMI (Infinity Fabric) mesh links bypass the PCIe bus
	// and CPU root complex entirely, so xGMI links are immune to PCIe ACS Request Redirect issues.
	for _, p := range src.Peers {
		if p.TargetNodeID == dstNodeID {
			if !p.DirectP2PCapable {
				return false, p.Fabric, fmt.Sprintf("peer link reports non-capable: %s", p.Warning)
			}
			if p.Fabric == FabricXGMI {
				return true, FabricXGMI, "direct Infinity Fabric xGMI peer link"
			}
			// For PCIe peer links, check PCIe ACS Request Redirect conflict
			if e.cfg.EnforceACSZeroRedirect && ((src.ACSEnabled && src.ACSRedirect) || (dst.ACSEnabled && dst.ACSRedirect)) {
				return false, FabricPCIeSwitch, "PCIe Access Control Services (ACS) Request Redirect is active: peer TLPs will be redirected and dropped by CPU root complex"
			}
			return true, p.Fabric, ""
		}
	}

	// Check ACS Request Redirect on general PCIe path
	if e.cfg.EnforceACSZeroRedirect && ((src.ACSEnabled && src.ACSRedirect) || (dst.ACSEnabled && dst.ACSRedirect)) {
		return false, FabricPCIeSwitch, "PCIe Access Control Services (ACS) Request Redirect is active: peer TLPs will be redirected and dropped by CPU root complex"
	}

	// Default fallback to PCIe host bridge if both share NUMA domain
	if src.NUMANode == dst.NUMANode {
		return true, FabricPCIeHostBridge, "fallback PCIe Host Bridge (same NUMA node)"
	}

	return false, FabricNone, fmt.Sprintf("nodes on differing NUMA nodes (%d vs %d) without explicit peer fabric", src.NUMANode, dst.NUMANode)
}

// ExportVRAMToDMABUF exports a GPU VRAM allocation into a Linux DMA-BUF file descriptor.
// Emulates AMDKFD_IOC_EXPORT_DMABUF / DRM PRIME dma-buf export.
func (e *AMDGPUDirectHAL) ExportVRAMToDMABUF(nodeID int, vaddr uintptr, size uint64) (*DMABUFHandle, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	node, ok := e.nodes[nodeID]
	if !ok {
		return nil, fmt.Errorf("amddirect: unknown node ID %d", nodeID)
	}
	if !node.DMABUFCapable {
		return nil, fmt.Errorf("amddirect: node %d does not support kernel DMA-BUF export", nodeID)
	}
	if size == 0 {
		return nil, errors.New("amddirect: export size must be > 0")
	}

	// Page align with zero-safe fallback
	align := e.cfg.DefaultPageSize
	if align == 0 {
		align = 4096
	}
	alignedSize := ((size + align - 1) / align) * align

	e.nextFD++
	handle := &DMABUFHandle{
		FD:           e.nextFD,
		Size:         size,
		Offset:       0,
		AlignedSize:  alignedSize,
		VRAMAddress:  vaddr,
		NodeID:       nodeID,
		ExportOffset: 0,
		Closed:       false,
	}

	if e.cfg.EnableLargeBARCheck && !node.IsLargeBAR && size > SmallBARThresholdBytes {
		// Note warning for small BAR sliding window
	}

	e.dmabufs[handle.FD] = handle
	return handle, nil
}

// GetDMABUF retrieves an exported DMA-BUF handle by file descriptor.
func (e *AMDGPUDirectHAL) GetDMABUF(fd int) *DMABUFHandle {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.dmabufs[fd]
}

// CloseDMABUF releases an exported DMA-BUF handle.
func (e *AMDGPUDirectHAL) CloseDMABUF(fd int) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	handle, ok := e.dmabufs[fd]
	if !ok {
		return fmt.Errorf("amddirect: unknown dmabuf fd %d", fd)
	}
	handle.Closed = true
	delete(e.dmabufs, fd)
	return nil
}

// RegisterDMABUFForRDMA registers an exported DMA-BUF with the InfiniBand/RoCE verbs subsystem.
// Emulates ibv_reg_dmabuf_mr from ROCm-RDMA and RCCL net_ib_rocm.cc.
// Generates direct ScatterGatherElements (SGEs) pointing to GPU BAR1 VRAM, guaranteeing 0 staging copies.
// Chunking is applied if length exceeds 32-bit MaxSGELengthBytes to prevent integer truncation.
func (e *AMDGPUDirectHAL) RegisterDMABUFForRDMA(dmabufFD int, length uint64) (*RDMARegisteredRegion, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	handle, ok := e.dmabufs[dmabufFD]
	if !ok || handle.Closed {
		return nil, fmt.Errorf("amddirect: invalid or closed dmabuf fd %d", dmabufFD)
	}
	if length == 0 || length > handle.AlignedSize {
		length = handle.AlignedSize
	}

	e.nextKey++
	rkey := e.nextKey
	lkey := rkey + 1

	// Produce ScatterGatherElements chunked into <= MaxSGELengthBytes spans
	sges := make([]ScatterGatherElement, 0, (length+MaxSGELengthBytes-1)/MaxSGELengthBytes)
	remaining := length
	currAddr := handle.VRAMAddress
	for remaining > 0 {
		chunk := remaining
		if chunk > MaxSGELengthBytes {
			chunk = MaxSGELengthBytes
		}
		sges = append(sges, ScatterGatherElement{
			Address: currAddr,
			Length:  uint32(chunk),
			LKey:    lkey,
		})
		currAddr += uintptr(chunk)
		remaining -= chunk
	}

	region := &RDMARegisteredRegion{
		RKey:        rkey,
		LKey:        lkey,
		IOVA:        uint64(handle.VRAMAddress),
		Length:      length,
		DMABUFFD:    dmabufFD,
		NodeID:      handle.NodeID,
		SGEs:        sges,
		StagingCopy: 0, // Invariant: zero CPU staging copies
		Active:      true,
	}

	e.rdmaMRs[rkey] = region
	return region, nil
}

// DeregisterRDMARegion releases an RDMA memory registration.
func (e *AMDGPUDirectHAL) DeregisterRDMARegion(rkey uint32) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	region, ok := e.rdmaMRs[rkey]
	if !ok {
		return fmt.Errorf("amddirect: unknown RDMA rkey 0x%x", rkey)
	}
	region.Active = false
	delete(e.rdmaMRs, rkey)
	return nil
}

// GetRDMARegion retrieves an active RDMA memory region by its registration key (rkey).
func (e *AMDGPUDirectHAL) GetRDMARegion(rkey uint32) *RDMARegisteredRegion {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.rdmaMRs[rkey]
}

// RegisterRDMARegion registers an externally-created RDMARegisteredRegion with the HAL coordinator.
func (e *AMDGPUDirectHAL) RegisterRDMARegion(region *RDMARegisteredRegion) error {
	if region == nil {
		return errors.New("amddirect: cannot register nil RDMA region")
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	e.rdmaMRs[region.RKey] = region
	return nil
}

// CreateQueuePair allocates and registers an RDMA Queue Pair in this coordinator.
func (e *AMDGPUDirectHAL) CreateQueuePair(initAttr QPInitAttr) (*RDMAQueuePair, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	e.nextQPN++
	qpNum := e.nextQPN
	qp, err := NewRDMAQueuePair(qpNum, initAttr)
	if err != nil {
		return nil, err
	}

	e.queuePairs[qpNum] = qp
	return qp, nil
}

// GetQueuePair retrieves an active Queue Pair by its QP number.
func (e *AMDGPUDirectHAL) GetQueuePair(qpNum uint32) *RDMAQueuePair {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.queuePairs[qpNum]
}

// DestroyQueuePair destroys an active Queue Pair and flushes its remaining work.
func (e *AMDGPUDirectHAL) DestroyQueuePair(qpNum uint32) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	qp, ok := e.queuePairs[qpNum]
	if !ok {
		return fmt.Errorf("amddirect: unknown queue pair number %d", qpNum)
	}

	_ = qp.Modify(QPAttr{State: QPStateError})
	delete(e.queuePairs, qpNum)
	return nil
}

// ExecuteNVMeP2PTransfer executes a direct NVMe-to-GPU peer-to-peer DMA transfer (BaM / SPDK-lite).
// The NVMe controller DMA engine reads or writes flash blocks directly to/from the GPU VRAM address
// over the PCIe bus without intermediate host DRAM staging copies.
func (e *AMDGPUDirectHAL) ExecuteNVMeP2PTransfer(cmd *NVMeP2PCommand) error {
	if cmd == nil {
		return errors.New("amddirect: nil NVMe P2P command")
	}
	if cmd.ByteLength == 0 {
		return errors.New("amddirect: transfer byte length must be > 0")
	}
	if cmd.TargetVRAMAddr == 0 {
		return errors.New("amddirect: target VRAM address cannot be 0")
	}
	if cmd.Opcode != 0 && cmd.Opcode != NVMeOpcodeRead && cmd.Opcode != NVMeOpcodeWrite {
		return fmt.Errorf("amddirect: unsupported NVMe opcode 0x%02x (must be 0x01 Write or 0x02 Read)", cmd.Opcode)
	}

	start := time.Now()
	// Simulate controller direct DMA execution over PCIe P2P bus
	cmd.Completed = true
	cmd.Status = 0
	cmd.DurationNanos = time.Since(start).Nanoseconds()

	atomic.AddInt64(&e.transfers, 1)
	atomic.AddUint64(&e.bytesMoved, cmd.ByteLength)
	return nil
}

// ExecuteNVMeP2PTransferAsync initiates an asynchronous direct NVMe-to-GPU peer-to-peer DMA transfer.
// Returns a receive-only channel that receives the completion error (or nil on success) once the transfer finishes.
func (e *AMDGPUDirectHAL) ExecuteNVMeP2PTransferAsync(cmd *NVMeP2PCommand) (<-chan error, error) {
	if cmd == nil {
		return nil, errors.New("amddirect: nil NVMe P2P command")
	}
	if cmd.ByteLength == 0 {
		return nil, errors.New("amddirect: transfer byte length must be > 0")
	}
	if cmd.TargetVRAMAddr == 0 {
		return nil, errors.New("amddirect: target VRAM address cannot be 0")
	}
	if cmd.Opcode != 0 && cmd.Opcode != NVMeOpcodeRead && cmd.Opcode != NVMeOpcodeWrite {
		return nil, fmt.Errorf("amddirect: unsupported NVMe opcode 0x%02x (must be 0x01 Write or 0x02 Read)", cmd.Opcode)
	}

	done := make(chan error, 1)
	go func() {
		err := e.ExecuteNVMeP2PTransfer(cmd)
		done <- err
	}()
	return done, nil
}

// TransferP2P performs a validated peer-to-peer data transfer between two AMD GPU nodes.
func (e *AMDGPUDirectHAL) TransferP2P(srcNodeID, dstNodeID int, size uint64) (AMDFabricType, float64, error) {
	ok, fabric, reason := e.ValidateP2PRoute(srcNodeID, dstNodeID)
	if !ok {
		return FabricNone, 0, fmt.Errorf("amddirect: P2P transfer route refused: %s", reason)
	}

	e.mu.RLock()
	src := e.nodes[srcNodeID]
	var bwidth float64 = 32.0 // default Gen4 fallback
	for _, p := range src.Peers {
		if p.TargetNodeID == dstNodeID {
			bwidth = p.BandwidthGBps
			break
		}
	}
	e.mu.RUnlock()

	atomic.AddInt64(&e.transfers, 1)
	atomic.AddUint64(&e.bytesMoved, size)
	return fabric, bwidth, nil
}

// RegisterSignal registers an HSA memory signal with the coordinator.
func (e *AMDGPUDirectHAL) RegisterSignal(s *HSAMemorySignal) {
	if s == nil {
		return
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	e.signals[s.SignalID] = s
}

// GetSignal retrieves a registered HSA memory signal by ID.
func (e *AMDGPUDirectHAL) GetSignal(id string) *HSAMemorySignal {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.signals[id]
}

// RegisterDoorbell registers an HSA dispatch doorbell with the coordinator.
func (e *AMDGPUDirectHAL) RegisterDoorbell(d *HSADoorbell) {
	if d == nil {
		return
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	e.doorbells[d.ID] = d
}

// GetDoorbell retrieves a registered HSA dispatch doorbell by ID.
func (e *AMDGPUDirectHAL) GetDoorbell(id string) *HSADoorbell {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.doorbells[id]
}

// AuditReport summarizes the configuration, hardware posture, and potential bottlenecks.
type AuditReport struct {
	TotalNodes          int      `json:"total_nodes"`
	NodesWithLargeBAR   int      `json:"nodes_with_large_bar"`
	NodesWithSmallBAR   int      `json:"nodes_with_small_bar"`
	ACSConflictDetected bool     `json:"acs_conflict_detected"`
	ActiveDMABUFCount   int      `json:"active_dmabuf_count"`
	ActiveRDMARegions   int      `json:"active_rdma_regions"`
	ActiveQueuePairs    int      `json:"active_queue_pairs"`
	TotalTransfers      int64    `json:"total_transfers"`
	TotalBytesMoved     uint64   `json:"total_bytes_moved"`
	Warnings            []string `json:"warnings"`
	Healthy             bool     `json:"healthy"`
}

// Audit runs diagnostic checks over the registered topology and active allocations.
func (e *AMDGPUDirectHAL) Audit() AuditReport {
	e.mu.RLock()
	defer e.mu.RUnlock()

	rep := AuditReport{
		TotalNodes:        len(e.nodes),
		ActiveDMABUFCount: len(e.dmabufs),
		ActiveRDMARegions: len(e.rdmaMRs),
		ActiveQueuePairs:  len(e.queuePairs),
		TotalTransfers:    atomic.LoadInt64(&e.transfers),
		TotalBytesMoved:   atomic.LoadUint64(&e.bytesMoved),
		Warnings:          make([]string, 0),
		Healthy:           true,
	}

	for _, n := range e.nodes {
		if n.IsLargeBAR {
			rep.NodesWithLargeBAR++
		} else {
			rep.NodesWithSmallBAR++
			if e.cfg.EnableLargeBARCheck {
				rep.Warnings = append(rep.Warnings, fmt.Sprintf("node %d (%s) running with 256 MiB Small BAR; P2P/RDMA throughput degraded", n.NodeID, n.DeviceName))
			}
		}

		if n.ACSEnabled && n.ACSRedirect {
			rep.ACSConflictDetected = true
			if e.cfg.EnforceACSZeroRedirect {
				rep.Healthy = false
				rep.Warnings = append(rep.Warnings, fmt.Sprintf("node %d (%s) has PCIe ACS Request Redirect enabled; peer P2P transactions will fail", n.NodeID, n.DeviceName))
			}
		}
	}

	return rep
}

// JSON encodes the AuditReport as indented JSON bytes.
func (r AuditReport) JSON() ([]byte, error) {
	return json.MarshalIndent(r, "", "  ")
}
