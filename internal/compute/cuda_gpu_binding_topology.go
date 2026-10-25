package compute

import (
	"fmt"
	"strconv"
	"strings"
)

// GPUPnPHop is one ancestor of a GPU in the device tree, nearest first: on
// Windows the PnP parent chain up to the first non-PCI ancestor (the root
// complex); on Linux the sysfs PCI bridges.
type GPUPnPHop struct {
	InstanceID string `json:"instance_id,omitempty"`
	Name       string `json:"name,omitempty"`
	PCIAddress string `json:"pci_address,omitempty"`
	ACPIPath   string `json:"acpi_path,omitempty"`
}

// GPUUSB4HostRouter is a USB4 host router (Windows PCI\USB4_MS_CM device). It
// is a sibling of the tunnel root ports, not a PnP ancestor of the GPU.
type GPUUSB4HostRouter GPUPnPHop

type winPnPHopRaw struct {
	InstanceID    string    `json:"instance_id"`
	FriendlyName  string    `json:"DEVPKEY_Device_FriendlyName"`
	DeviceDesc    string    `json:"DEVPKEY_Device_DeviceDesc"`
	BusNumber     *uint32   `json:"DEVPKEY_Device_BusNumber"`
	Address       *uint32   `json:"DEVPKEY_Device_Address"`
	LocationPaths psStrings `json:"DEVPKEY_Device_LocationPaths"`
}

func isPnPPCIInstance(id string) bool {
	return strings.HasPrefix(strings.ToUpper(id), `PCI\`)
}

func (r winPnPHopRaw) hop() GPUPnPHop {
	h := GPUPnPHop{InstanceID: r.InstanceID, Name: firstNonEmpty(r.FriendlyName, r.DeviceDesc)}
	if isPnPPCIInstance(r.InstanceID) && r.BusNumber != nil && r.Address != nil {
		h.PCIAddress = windowsPCIAddress(r.BusNumber, r.Address, "")
	}
	for _, p := range r.LocationPaths {
		if strings.HasPrefix(strings.ToUpper(p), "ACPI(") {
			h.ACPIPath = p
			break
		}
	}
	return h
}

// windowsParentChain converts the queried PnP parents and returns the PCI
// address of the topmost PCI hop: the root port the GPU's link hangs off.
func windowsParentChain(raw []winPnPHopRaw) ([]GPUPnPHop, string) {
	var chain []GPUPnPHop
	rootPort := ""
	for _, r := range raw {
		h := r.hop()
		chain = append(chain, h)
		if isPnPPCIInstance(h.InstanceID) {
			rootPort = h.PCIAddress
		}
	}
	return chain, rootPort
}

// linuxParentChain lists a resolved sysfs device path's PCI ancestors nearest first.
func linuxParentChain(resolved string) ([]GPUPnPHop, string) {
	rootPort, ancestry := PCIAncestryFromDevicePath(resolved)
	chain := make([]GPUPnPHop, 0, len(ancestry))
	for i := len(ancestry) - 1; i >= 0; i-- {
		chain = append(chain, GPUPnPHop{PCIAddress: ancestry[i]})
	}
	return chain, rootPort
}

// acpiTailIndex returns n when an ACPI path's last node is <prefix><n>, e.g.
// "ACPI(_SB_)#ACPI(PC00)#ACPI(TRP3)" gives 3 for "TRP".
func acpiTailIndex(path, prefix string) (int, bool) {
	nodes := strings.Split(strings.TrimSpace(path), "#")
	last := nodes[len(nodes)-1]
	if !strings.HasPrefix(last, "ACPI(") || !strings.HasSuffix(last, ")") {
		return 0, false
	}
	name := strings.TrimSuffix(strings.TrimPrefix(last, "ACPI("), ")")
	if !strings.HasPrefix(name, prefix) {
		return 0, false
	}
	n, err := strconv.Atoi(strings.TrimPrefix(name, prefix))
	return n, err == nil
}

// associateUSB4HostRouter links a GPU to the USB4 host router that owns its
// tunnel root port. Intel firmware names tunnel root ports TRP<n> and host
// routers TDM<m>, two root ports per router (m = n/2); without both names no
// router is claimed, and the basis says which names decided it.
func associateUSB4HostRouter(dev *GPUBindingDevice, routers []GPUUSB4HostRouter) {
	rootACPI := ""
	for i := len(dev.ParentChain) - 1; i >= 0; i-- {
		if isPnPPCIInstance(dev.ParentChain[i].InstanceID) {
			rootACPI = dev.ParentChain[i].ACPIPath
			break
		}
	}
	n, ok := acpiTailIndex(rootACPI, "TRP")
	if !ok {
		return
	}
	for _, r := range routers {
		if m, ok := acpiTailIndex(r.ACPIPath, "TDM"); ok && m == n/2 {
			dev.USB4HostRouter = r.InstanceID
			dev.USB4RouterBasis = fmt.Sprintf("acpi TRP%d -> TDM%d", n, m)
			return
		}
	}
}
