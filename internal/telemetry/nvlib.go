// Copyright (c) 2021, 2022, NVIDIA CORPORATION. All rights reserved.
// Ported: sysfs PCI device discovery with its vendor and class constants, and
// the MIG profile string grammar with its memory-size rounding. Changes were made.
//
// What fak adds: every read takes an explicit sysfs root so it runs against a
// fixture tree, and discovery is fail-closed. A missing root or a host with no
// NVIDIA device is ErrNVLibUnavailable, never an empty list that reads as
// "nothing wrong". Not ported: Mellanox devices, BAR resources, SR-IOV and the
// pci.ids name database.

package telemetry

import (
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// ErrNVLibUnavailable is returned when the sysfs root is missing or holds no
// NVIDIA device of the requested kind.
var ErrNVLibUnavailable = errors.New("telemetry: nvidia pci devices unavailable")

// ErrMIGProfileInvalid is returned by ParseMIGProfile for a string outside the
// MIG profile grammar.
var ErrMIGProfileInvalid = errors.New("telemetry: invalid mig profile")

// PCI identifiers, verbatim from upstream nvpci.
const (
	PCIDevicesRoot               = "/sys/bus/pci/devices"
	PCINvidiaVendorID     uint16 = 0x10de
	PCIVgaControllerClass uint32 = 0x030000
	PCI3dControllerClass  uint32 = 0x030200
	PCINvSwitchClass      uint32 = 0x068000
)

// NvidiaPCIDevice is one NVIDIA PCI function read from sysfs.
type NvidiaPCIDevice struct {
	Path            string `json:"path"`
	Address         string `json:"address"`
	Vendor          uint16 `json:"vendor"`
	Class           uint32 `json:"class"`
	Device          uint16 `json:"device"`
	SubsystemVendor uint16 `json:"subsystem_vendor"`
	SubsystemDevice uint16 `json:"subsystem_device"`
	Driver          string `json:"driver,omitempty"`
	IommuGroup      int    `json:"iommu_group"`
	NumaNode        int    `json:"numa_node"`
}

// IsVGAController reports class 0x030000.
func (d NvidiaPCIDevice) IsVGAController() bool { return d.Class == PCIVgaControllerClass }

// Is3DController reports class 0x030200.
func (d NvidiaPCIDevice) Is3DController() bool { return d.Class == PCI3dControllerClass }

// IsNVSwitch reports class 0x068000.
func (d NvidiaPCIDevice) IsNVSwitch() bool { return d.Class == PCINvSwitchClass }

// IsGPU is a VGA controller (older boards) or a 3D controller (newer boards).
func (d NvidiaPCIDevice) IsGPU() bool { return d.IsVGAController() || d.Is3DController() }

// DiscoverNvidiaPCI returns every NVIDIA PCI function under root (normally
// PCIDevicesRoot), sorted by bus address. A missing root or no NVIDIA function
// is ErrNVLibUnavailable; a malformed NVIDIA entry is an error naming it.
func DiscoverNvidiaPCI(root string) ([]NvidiaPCIDevice, error) {
	entries, err := os.ReadDir(root)
	if err != nil {
		return nil, fmt.Errorf("%w: read %s: %w", ErrNVLibUnavailable, root, err)
	}
	var out []NvidiaPCIDevice
	for _, e := range entries {
		dev, ok, err := readNvidiaPCIDevice(filepath.Join(root, e.Name()), e.Name())
		if err != nil {
			return nil, fmt.Errorf("pci device %s: %w", e.Name(), err)
		}
		if ok {
			out = append(out, dev)
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("%w: no NVIDIA devices under %s", ErrNVLibUnavailable, root)
	}
	sort.Slice(out, func(i, j int) bool { return pciAddressID(out[i].Address) < pciAddressID(out[j].Address) })
	return out, nil
}

// DiscoverNvidiaGPUs is DiscoverNvidiaPCI filtered to GPUs, refusing with
// ErrNVLibUnavailable when only non-GPU functions such as NVSwitches exist.
func DiscoverNvidiaGPUs(root string) ([]NvidiaPCIDevice, error) {
	all, err := DiscoverNvidiaPCI(root)
	if err != nil {
		return nil, err
	}
	var gpus []NvidiaPCIDevice
	for _, d := range all {
		if d.IsGPU() {
			gpus = append(gpus, d)
		}
	}
	if len(gpus) == 0 {
		return nil, fmt.Errorf("%w: no NVIDIA GPUs under %s", ErrNVLibUnavailable, root)
	}
	return gpus, nil
}

func pciAddressID(address string) uint64 {
	address = strings.ReplaceAll(address, ":", "")
	address = strings.ReplaceAll(address, ".", "")
	id, _ := strconv.ParseUint(address, 16, 64)
	return id
}

func readPCIFieldString(devicePath, field string) (string, error) {
	raw, err := os.ReadFile(filepath.Join(devicePath, field))
	if err != nil {
		return "", fmt.Errorf("read %s: %w", field, err)
	}
	return strings.TrimSpace(string(raw)), nil
}

func readPCIField(devicePath, field string, bitSize int) (uint64, error) {
	s, err := readPCIFieldString(devicePath, field)
	if err != nil {
		return 0, err
	}
	v, err := strconv.ParseUint(s, 0, bitSize)
	if err != nil {
		return 0, fmt.Errorf("parse %s: %w", field, err)
	}
	return v, nil
}

func readNvidiaPCIDevice(devicePath, address string) (NvidiaPCIDevice, bool, error) {
	vendor, err := readPCIField(devicePath, "vendor", 16)
	if err != nil {
		return NvidiaPCIDevice{}, false, err
	}
	if uint16(vendor) != PCINvidiaVendorID {
		return NvidiaPCIDevice{}, false, nil
	}
	class, err := readPCIField(devicePath, "class", 32)
	if err != nil {
		return NvidiaPCIDevice{}, false, err
	}
	device, err := readPCIField(devicePath, "device", 16)
	if err != nil {
		return NvidiaPCIDevice{}, false, err
	}
	numaStr, err := readPCIFieldString(devicePath, "numa_node")
	if err != nil {
		return NvidiaPCIDevice{}, false, err
	}
	// Signed: -1 means no NUMA affinity.
	numa, err := strconv.ParseInt(numaStr, 0, 64)
	if err != nil {
		return NvidiaPCIDevice{}, false, fmt.Errorf("parse numa_node: %w", err)
	}
	// Some virtualised and passthrough topologies omit the subsystem files.
	subVendor, err := readPCIField(devicePath, "subsystem_vendor", 16)
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return NvidiaPCIDevice{}, false, err
	}
	subDevice, err := readPCIField(devicePath, "subsystem_device", 16)
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return NvidiaPCIDevice{}, false, err
	}
	driver, err := pciDriver(devicePath)
	if err != nil {
		return NvidiaPCIDevice{}, false, fmt.Errorf("driver: %w", err)
	}
	group, err := pciIOMMUGroup(devicePath)
	if err != nil {
		return NvidiaPCIDevice{}, false, fmt.Errorf("iommu_group: %w", err)
	}
	return NvidiaPCIDevice{
		Path:            devicePath,
		Address:         address,
		Vendor:          uint16(vendor),
		Class:           uint32(class),
		Device:          uint16(device),
		SubsystemVendor: uint16(subVendor),
		SubsystemDevice: uint16(subDevice),
		Driver:          driver,
		IommuGroup:      int(group),
		NumaNode:        int(numa),
	}, true, nil
}

func pciDriver(devicePath string) (string, error) {
	target, err := filepath.EvalSymlinks(filepath.Join(devicePath, "driver"))
	switch {
	case os.IsNotExist(err):
		return "", nil
	case err == nil:
		return filepath.Base(target), nil
	}
	return "", err
}

func pciIOMMUGroup(devicePath string) (int64, error) {
	target, err := filepath.EvalSymlinks(filepath.Join(devicePath, "iommu_group"))
	switch {
	case os.IsNotExist(err):
		return -1, nil
	case err == nil:
		s := strings.TrimSpace(filepath.Base(target))
		g, err := strconv.ParseInt(s, 0, 64)
		if err != nil {
			return 0, fmt.Errorf("parse %q: %w", s, err)
		}
		return g, nil
	}
	return 0, err
}

// MIG profile attributes, verbatim from upstream.
const (
	MIGAttributeMediaExtensions    = "me"
	MIGAttributeMediaExtensionsAll = "me.all"
	MIGAttributeGraphics           = "gfx"
)

// MIGProfile is a parsed MIG profile string such as "1g.5gb", "1c.2g.10gb" or
// "1g.10gb+me". C equals G when the string has no compute-slice field.
type MIGProfile struct {
	C             int      `json:"c"`
	G             int      `json:"g"`
	GB            int      `json:"gb"`
	Attributes    []string `json:"attributes,omitempty"`
	NegAttributes []string `json:"neg_attributes,omitempty"`
}

// String renders the profile in the same grammar ParseMIGProfile accepts.
func (p MIGProfile) String() string {
	var suffix string
	if len(p.Attributes) > 0 {
		suffix = "+" + strings.Join(p.Attributes, ",")
	}
	if len(p.NegAttributes) > 0 {
		suffix = "-" + strings.Join(p.NegAttributes, ",")
	}
	if p.C == p.G {
		return fmt.Sprintf("%dg.%dgb%s", p.G, p.GB, suffix)
	}
	return fmt.Sprintf("%dc.%dg.%dgb%s", p.C, p.G, p.GB, suffix)
}

// ParseMIGProfile parses "[<c>c.]<g>g.<gb>gb[+attr,...|-attr,...]". Every
// rejection wraps ErrMIGProfileInvalid.
func ParseMIGProfile(profile string) (MIGProfile, error) {
	p, err := parseMIGProfile(profile)
	if err != nil {
		return MIGProfile{}, fmt.Errorf("%w: %q: %s", ErrMIGProfileInvalid, profile, err)
	}
	return p, nil
}

func parseMIGProfile(profile string) (MIGProfile, error) {
	if profile == "" {
		return MIGProfile{}, errors.New("profile is the empty string")
	}
	split := strings.SplitN(profile, "+", 2)
	negsplit := strings.SplitN(profile, "-", 2)
	if len(split) == 2 && len(negsplit) == 2 {
		return MIGProfile{}, errors.New("contains both '+/-' attributes")
	}
	if len(split) == 1 {
		split = negsplit
	}
	c, g, gb, err := parseMIGProfileFields(split[0])
	if err != nil {
		return MIGProfile{}, err
	}
	p := MIGProfile{C: c, G: g, GB: gb}
	if len(split) == 1 {
		return p, nil
	}
	attrs, err := parseMIGProfileAttributes(split[1])
	if err != nil {
		return MIGProfile{}, err
	}
	if len(negsplit) == 2 {
		p.NegAttributes = attrs
		return p, nil
	}
	p.Attributes = attrs
	return p, nil
}

func parseMIGProfileField(s, field string) (int, error) {
	if strings.TrimSpace(s) != s {
		return -1, fmt.Errorf("leading or trailing spaces on '%%d%s'", field)
	}
	if !strings.HasSuffix(s, field) {
		return -1, fmt.Errorf("missing '%s' from '%%d%s'", field, field)
	}
	v, err := strconv.Atoi(strings.TrimSuffix(s, field))
	if err != nil {
		return -1, fmt.Errorf("malformed number in '%%d%s'", field)
	}
	return v, nil
}

func parseMIGProfileFields(s string) (c, g, gb int, err error) {
	split := strings.SplitN(s, ".", 3)
	switch len(split) {
	case 3:
		if c, err = parseMIGProfileField(split[0], "c"); err != nil {
			return -1, -1, -1, err
		}
		if g, err = parseMIGProfileField(split[1], "g"); err != nil {
			return -1, -1, -1, err
		}
		if gb, err = parseMIGProfileField(split[2], "gb"); err != nil {
			return -1, -1, -1, err
		}
		return c, g, gb, nil
	case 2:
		if g, err = parseMIGProfileField(split[0], "g"); err != nil {
			return -1, -1, -1, err
		}
		if gb, err = parseMIGProfileField(split[1], "gb"); err != nil {
			return -1, -1, -1, err
		}
		return g, g, gb, nil
	}
	return -1, -1, -1, errors.New("parsed wrong number of fields, expected 2 or 3")
}

func parseMIGProfileAttributes(s string) ([]string, error) {
	attrs := strings.Split(s, ",")
	seen := make(map[string]bool, len(attrs))
	for _, a := range attrs {
		if seen[a] {
			return nil, errors.New("non unique attribute in list")
		}
		if a == "" {
			return nil, errors.New("empty attribute in list")
		}
		if strings.TrimSpace(a) != a {
			return nil, errors.New("leading or trailing spaces in attribute")
		}
		if a[0] >= '0' && a[0] <= '9' {
			return nil, errors.New("attribute begins with a number")
		}
		if a[0] == '.' || a[len(a)-1] == '.' {
			return nil, errors.New("attribute begins/ends with a dot")
		}
		for _, r := range a {
			if (r < 'a' || r > 'z') && (r < 'A' || r > 'Z') && (r < '0' || r > '9') && r != '.' {
				return nil, errors.New("non alpha-numeric character or digit in attribute")
			}
		}
		seen[a] = true
	}
	return attrs, nil
}

// MIGMemorySizeGB is the GB figure a MIG profile name carries for an instance
// of migMemorySizeMB on a device of totalDeviceMemoryBytes. It rounds the
// instance's share of the device up to eighths, then scales by the device size
// in whole GiB, which is why a 4864 MiB slice of an 80 GB board names "10gb".
func MIGMemorySizeGB(totalDeviceMemoryBytes, migMemorySizeMB uint64) uint64 {
	const fracDenominator = 8
	const oneMB = 1024 * 1024
	const oneGB = 1024 * 1024 * 1024
	if totalDeviceMemoryBytes == 0 {
		return 0
	}
	frac := (float64(migMemorySizeMB) * oneMB) / float64(totalDeviceMemoryBytes)
	frac = math.Ceil(frac*fracDenominator) / fracDenominator
	totalGB := float64((totalDeviceMemoryBytes + oneGB - 1) / oneGB)
	return uint64(math.Round(frac * totalGB))
}
