package gpubinding

import (
	"fmt"
	"path/filepath"
	"strconv"
	"strings"
)

// PnPHop is one ancestor of a GPU in the device tree, nearest first: on
// Windows the PnP parent chain up to the first non-PCI ancestor (the root
// complex); on Linux the sysfs PCI bridges.
type PnPHop struct {
	InstanceID string `json:"instance_id,omitempty"`
	Name       string `json:"name,omitempty"`
	PCIAddress string `json:"pci_address,omitempty"`
	ACPIPath   string `json:"acpi_path,omitempty"`
}

// USB4HostRouter is a USB4 host router (Windows PCI\USB4_MS_CM device). It
// is a sibling of the tunnel root ports, not a PnP ancestor of the GPU.
type USB4HostRouter PnPHop

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

func (r winPnPHopRaw) hop() PnPHop {
	h := PnPHop{InstanceID: r.InstanceID, Name: firstNonEmpty(r.FriendlyName, r.DeviceDesc)}
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
func windowsParentChain(raw []winPnPHopRaw) ([]PnPHop, string) {
	var chain []PnPHop
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
func linuxParentChain(resolved string) ([]PnPHop, string) {
	rootPort, ancestry := pciAncestryFromDevicePath(resolved)
	chain := make([]PnPHop, 0, len(ancestry))
	for i := len(ancestry) - 1; i >= 0; i-- {
		chain = append(chain, PnPHop{PCIAddress: ancestry[i]})
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
func associateUSB4HostRouter(dev *Device, routers []USB4HostRouter) {
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

// isPCIAddressComponent reports whether a path element is a PCI BDF
// (DDDD:BB:DD.F). Underscores stand in for colons on filesystems that forbid them.
func isPCIAddressComponent(elem string) bool {
	parts := strings.Split(strings.ReplaceAll(elem, "_", ":"), ":")
	if len(parts) != 3 || len(parts[0]) != 4 || len(parts[1]) != 2 {
		return false
	}
	devFn := strings.Split(parts[2], ".")
	if len(devFn) != 2 || len(devFn[0]) != 2 || len(devFn[1]) != 1 {
		return false
	}
	for _, h := range []string{parts[0], parts[1], devFn[0], devFn[1]} {
		if _, err := strconv.ParseUint(h, 16, 16); err != nil {
			return false
		}
	}
	return true
}

// isPCIHostBridgeComponent reports whether a path element is a host bridge
// directory (pciDDDD:BB).
func isPCIHostBridgeComponent(elem string) bool {
	if !strings.HasPrefix(elem, "pci") {
		return false
	}
	return isPCIAddressComponent(strings.TrimPrefix(elem, "pci") + ":00.0")
}

// pciAncestryFromDevicePath extracts the PCIe ancestry from a resolved sysfs
// device path such as /sys/devices/pci0000:00/0000:00:01.1/0000:01:00.0/0000:02:00.0.
// It returns the root port and the chain of upstream bridge addresses (root
// port first, the device itself excluded). A path with no bridge above the
// device (a root-complex integrated endpoint) or no pciDDDD:BB host bridge
// returns an empty root port: ancestry is unknown, never assumed.
func pciAncestryFromDevicePath(resolved string) (rootPort string, ancestry []string) {
	elems := strings.FieldsFunc(filepath.ToSlash(resolved), func(r rune) bool { return r == '/' })
	host := -1
	for i, e := range elems {
		if isPCIHostBridgeComponent(e) {
			host = i
		}
	}
	if host < 0 {
		return "", nil
	}
	var chain []string
	for _, e := range elems[host+1:] {
		if !isPCIAddressComponent(e) {
			break
		}
		chain = append(chain, strings.ReplaceAll(e, "_", ":"))
	}
	if len(chain) < 2 {
		return "", nil
	}
	ancestry = chain[:len(chain)-1]
	return ancestry[0], ancestry
}

// pnpHardwareID extracts the 4-hex-digit value following key (e.g. "VEN_") in a
// PNP device ID, formatted like sysfs ("0x10de"), or "" when absent.
func pnpHardwareID(pnp, key string) string {
	i := strings.Index(strings.ToUpper(pnp), key)
	if i < 0 || len(pnp) < i+len(key)+4 {
		return ""
	}
	id := pnp[i+len(key) : i+len(key)+4]
	if _, err := strconv.ParseUint(id, 16, 16); err != nil {
		return ""
	}
	return "0x" + strings.ToLower(id)
}
