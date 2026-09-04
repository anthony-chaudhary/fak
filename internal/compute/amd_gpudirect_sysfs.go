// Package compute implements hardware abstraction, tensor computation, memory slab management,
// and zero-copy device interconnect acceleration for the fak agent kernel.
package compute

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// ErrSysfsUnavailable indicates that Linux KFD/DRM sysfs topology interfaces are not reachable.
var ErrSysfsUnavailable = errors.New("amddirect: Linux KFD sysfs topology (/sys/class/kfd/kfd/topology) is unavailable on this host")

// SysfsNodeProperties captures raw key-value properties parsed from a KFD node's properties file.
type SysfsNodeProperties struct {
	GPUID        int
	DeviceName   string
	LocationID   uint32
	VendorID     uint32
	DeviceID     uint32
	NumSIMD      int
	MaxWavefront int
	NumMemBanks  int
	NumIOLinks   int
}

// ParseKFDProperties parses a KFD sysfs properties file (key value space-separated).
func ParseKFDProperties(content string) SysfsNodeProperties {
	props := SysfsNodeProperties{}
	scanner := bufio.NewScanner(strings.NewReader(content))
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}
		key := fields[0]
		val := fields[1]

		switch key {
		case "gpu_id":
			props.GPUID, _ = strconv.Atoi(val)
		case "name":
			props.DeviceName = val
		case "location_id":
			v, _ := strconv.ParseUint(val, 0, 32)
			props.LocationID = uint32(v)
		case "vendor_id":
			v, _ := strconv.ParseUint(val, 0, 32)
			props.VendorID = uint32(v)
		case "device_id":
			v, _ := strconv.ParseUint(val, 0, 32)
			props.DeviceID = uint32(v)
		case "simd_count":
			props.NumSIMD, _ = strconv.Atoi(val)
		case "max_waves_per_simd":
			props.MaxWavefront, _ = strconv.Atoi(val)
		case "mem_banks_count":
			props.NumMemBanks, _ = strconv.Atoi(val)
		case "io_links_count":
			props.NumIOLinks, _ = strconv.Atoi(val)
		}
	}
	return props
}

// ParsePCIeResourceSizes parses a Linux sysfs PCI resource file to calculate BAR sizes.
// Each line corresponds to a PCIe BAR: [start, end, flags].
func ParsePCIeResourceSizes(content string) []uint64 {
	barSizes := make([]uint64, 0, 6)
	scanner := bufio.NewScanner(strings.NewReader(content))
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) < 2 {
			continue
		}
		start, err1 := strconv.ParseUint(fields[0], 0, 64)
		end, err2 := strconv.ParseUint(fields[1], 0, 64)
		if err1 == nil && err2 == nil && end >= start && start > 0 {
			barSizes = append(barSizes, end-start+1)
		} else {
			barSizes = append(barSizes, 0)
		}
	}
	return barSizes
}

// ProbeKFDTopology scans the sysfs KFD topology tree (under sysfsRoot) and returns discovered AMDDeviceNodes.
func ProbeKFDTopology(sysfsRoot string) ([]AMDDeviceNode, error) {
	if sysfsRoot == "" {
		sysfsRoot = "/sys"
	}

	topoDir := filepath.Join(sysfsRoot, "class", "kfd", "kfd", "topology", "nodes")
	entries, err := os.ReadDir(topoDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, ErrSysfsUnavailable
		}
		return nil, fmt.Errorf("amddirect: reading KFD topology directory: %w", err)
	}

	nodes := make([]AMDDeviceNode, 0, len(entries))
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		nodeID, err := strconv.Atoi(entry.Name())
		if err != nil {
			continue
		}

		nodePath := filepath.Join(topoDir, entry.Name())
		propFile := filepath.Join(nodePath, "properties")
		data, err := os.ReadFile(propFile)
		if err != nil {
			continue
		}

		props := ParseKFDProperties(string(data))
		// If gpu_id is 0, this is the host CPU node in KFD topology
		if props.GPUID == 0 {
			continue
		}

		// Probe VRAM size from mem_banks
		var totalVRAM uint64
		memBanksDir := filepath.Join(nodePath, "mem_banks")
		if bankEntries, err := os.ReadDir(memBanksDir); err == nil {
			for _, b := range bankEntries {
				bPropsData, err := os.ReadFile(filepath.Join(memBanksDir, b.Name(), "properties"))
				if err != nil {
					continue
				}
				bScanner := bufio.NewScanner(strings.NewReader(string(bPropsData)))
				var size uint64
				var heapType uint32
				for bScanner.Scan() {
					f := strings.Fields(bScanner.Text())
					if len(f) >= 2 {
						if f[0] == "size_in_bytes" {
							size, _ = strconv.ParseUint(f[1], 10, 64)
						} else if f[0] == "heap_type" {
							ht, _ := strconv.ParseUint(f[1], 10, 32)
							heapType = uint32(ht)
						}
					}
				}
				// heap_type 0 = LDS, 1 = GPU VRAM, 2 = System DRAM, 3 = GDS
				if heapType == 1 && size > totalVRAM {
					totalVRAM = size
				}
			}
		}

		// Form BDF from location_id
		bus := (props.LocationID >> 8) & 0xFF
		dev := (props.LocationID >> 3) & 0x1F
		fn := props.LocationID & 0x07
		bdf := fmt.Sprintf("0000:%02x:%02x.%d", bus, dev, fn)

		// Probe BAR sizes from PCI resource if available
		bar1Size := totalVRAM // Default assume Large BAR
		pciResFile := filepath.Join(sysfsRoot, "bus", "pci", "devices", bdf, "resource")
		if pciData, err := os.ReadFile(pciResFile); err == nil {
			barSizes := ParsePCIeResourceSizes(string(pciData))
			if len(barSizes) > 1 && barSizes[1] > 0 {
				bar1Size = barSizes[1]
			}
		}

		// Probe io_links for xGMI and PCIe peers
		peers := make([]PeerLink, 0)
		ioLinksDir := filepath.Join(nodePath, "io_links")
		if linkEntries, err := os.ReadDir(ioLinksDir); err == nil {
			for _, l := range linkEntries {
				lData, err := os.ReadFile(filepath.Join(ioLinksDir, l.Name(), "properties"))
				if err != nil {
					continue
				}
				lScanner := bufio.NewScanner(strings.NewReader(string(lData)))
				var linkType int
				var nodeTo int
				var minBw, maxBw uint64
				var minLat uint32
				for lScanner.Scan() {
					f := strings.Fields(lScanner.Text())
					if len(f) >= 2 {
						switch f[0] {
						case "type":
							linkType, _ = strconv.Atoi(f[1])
						case "node_to":
							nodeTo, _ = strconv.Atoi(f[1])
						case "min_bandwidth":
							minBw, _ = strconv.ParseUint(f[1], 10, 64)
						case "max_bandwidth":
							maxBw, _ = strconv.ParseUint(f[1], 10, 64)
						case "min_latency":
							ml, _ := strconv.ParseUint(f[1], 10, 32)
							minLat = uint32(ml)
						}
					}
				}

				fabric := FabricPCIeSwitch
				bwGBps := float64(maxBw) / 1024.0
				if bwGBps == 0 && minBw > 0 {
					bwGBps = float64(minBw) / 1024.0
				}
				if linkType == 2 { // 2 = xGMI
					fabric = FabricXGMI
					if bwGBps == 0 {
						bwGBps = 896.0 // standard Instinct MI300X xGMI
					}
					if minLat == 0 {
						minLat = 210 // ns
					}
				} else {
					if bwGBps == 0 {
						bwGBps = 64.0 // PCIe Gen5 x16
					}
					if minLat == 0 {
						minLat = 450 // ns
					}
				}

				peers = append(peers, PeerLink{
					TargetNodeID:     nodeTo,
					Fabric:           fabric,
					BandwidthGBps:    bwGBps,
					LatencyNanos:     minLat,
					DirectP2PCapable: true,
					Coherent:         fabric == FabricXGMI,
				})
			}
		}

		devNode := AMDDeviceNode{
			NodeID:         nodeID,
			GPUID:          props.GPUID,
			DeviceName:     props.DeviceName,
			Architecture:   "gfx942", // Default CDNA3/MI300X or RDNA3
			PCIeBDF:        bdf,
			NUMANode:       0,
			TotalVRAMBytes: totalVRAM,
			BAR1SizeBytes:  bar1Size,
			IsLargeBAR:     bar1Size >= totalVRAM,
			KeepVRAMMapped: true,
			DMABUFCapable:  true,
			Peers:          peers,
		}
		nodes = append(nodes, devNode)
	}

	return nodes, nil
}

type winDisplayInfo struct {
	Name   string `json:"name"`
	Driver string `json:"driver"`
	RAM    int64  `json:"ram"`
	PNP    string `json:"pnp"`
}

// ProbeWindowsDisplayTopology queries the Windows display subsystem for AMD GPUs via Win32_VideoController.
func ProbeWindowsDisplayTopology() ([]AMDDeviceNode, error) {
	if runtime.GOOS != "windows" {
		return nil, errors.New("amddirect: Windows display probe called on non-Windows host")
	}

	cmd := exec.Command("powershell", "-NoProfile", "-Command",
		"Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match 'Radeon|AMD' } | ForEach-Object { [pscustomobject]@{ name = $_.Name; driver = $_.DriverVersion; ram = [int64]$_.AdapterRAM; pnp = $_.PNPDeviceID } } | ConvertTo-Json -Compress")
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("amddirect: querying Win32_VideoController: %w", err)
	}

	trimmed := strings.TrimSpace(string(out))
	if trimmed == "" {
		return nil, ErrSysfsUnavailable
	}

	var items []winDisplayInfo
	if strings.HasPrefix(trimmed, "[") {
		if err := json.Unmarshal([]byte(trimmed), &items); err != nil {
			return nil, err
		}
	} else if strings.HasPrefix(trimmed, "{") {
		var single winDisplayInfo
		if err := json.Unmarshal([]byte(trimmed), &single); err != nil {
			return nil, err
		}
		items = append(items, single)
	} else {
		return nil, ErrSysfsUnavailable
	}

	if len(items) == 0 {
		return nil, ErrSysfsUnavailable
	}

	nodes := make([]AMDDeviceNode, 0, len(items))
	for idx, item := range items {
		arch := "gfx1103" // default APU (Phoenix / Strix)
		vram := uint64(item.RAM)
		bdf := "0000:7b:00.0"
		if strings.Contains(item.Name, "RX 7600") {
			arch = "gfx1102"
			vram = 8573157376 // 8 GiB physical VRAM
			bdf = "0000:03:00.0"
		} else if vram == 0 {
			vram = 2 * 1024 * 1024 * 1024
		} else if vram == 4293918720 { // WMI 4GB cap
			vram = 8 * 1024 * 1024 * 1024
		}

		node := AMDDeviceNode{
			NodeID:         idx,
			GPUID:          idx,
			DeviceName:     item.Name,
			Architecture:   arch,
			PCIeBDF:        bdf,
			NUMANode:       0,
			TotalVRAMBytes: vram,
			BAR1SizeBytes:  vram,
			IsLargeBAR:     true,
			KeepVRAMMapped: true,
			DMABUFCapable:  true,
		}
		nodes = append(nodes, node)
	}

	// Link peers across PCIe Host Bridge
	for i := range nodes {
		for j := range nodes {
			if i != j {
				nodes[i].Peers = append(nodes[i].Peers, PeerLink{
					TargetNodeID:     nodes[j].NodeID,
					Fabric:           FabricPCIeHostBridge,
					BandwidthGBps:    32.0,
					LatencyNanos:     650,
					DirectP2PCapable: true,
					Coherent:         false,
				})
			}
		}
	}

	return nodes, nil
}

// ProbeHostTopology auto-discovers physical AMD GPUs across Linux sysfs and Windows display devices.
func ProbeHostTopology(sysfsRoot string) ([]AMDDeviceNode, error) {
	if runtime.GOOS == "windows" {
		if nodes, err := ProbeWindowsDisplayTopology(); err == nil && len(nodes) > 0 {
			return nodes, nil
		}
	}
	return ProbeKFDTopology(sysfsRoot)
}
