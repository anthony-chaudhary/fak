package compute

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// GPUDriverPackage answers, for an extracted NVIDIA driver package, which
// package INF lists each present GPU's hardware ID and which one Windows would
// rank first, plus the ordered pnputil plan that would bind every covered GPU.
// The plan is printed, never run. Uncovered GPUs are ones no package INF lists:
// installing this package (and above all with -clean) leaves them unbound.
type GPUDriverPackage struct {
	Dir             string               `json:"dir"`
	DisplayINFs     int                  `json:"display_infs"`
	Devices         []GPUPackageDevice   `json:"devices"`
	PnputilCommands []string             `json:"pnputil_commands,omitempty"`
	Uncovered       []string             `json:"uncovered,omitempty"`
	Issues          []CUDADiscoveryIssue `json:"issues,omitempty"`
}

// GPUPackageDevice is one present GPU's coverage by the package.
type GPUPackageDevice struct {
	PCIAddress string   `json:"pci_address,omitempty"`
	InstanceID string   `json:"instance_id,omitempty"`
	ListedBy   []string `json:"listed_by,omitempty"`
	Chosen     string   `json:"chosen_inf,omitempty"`
	MatchedID  string   `json:"matched_id,omitempty"`
}

// hardwareIDRank orders INF matches the way Windows ranks a PCI device's IDs:
// VEN&DEV&SUBSYS&REV, VEN&DEV&SUBSYS, the class-code hardware IDs, then the
// compatible VEN&DEV&REV and VEN&DEV forms. Lower is preferred.
func hardwareIDRank(id string) int {
	u := strings.ToUpper(id)
	subsys, rev := strings.Contains(u, "&SUBSYS_"), strings.Contains(u, "&REV_")
	switch {
	case subsys && rev:
		return 0
	case subsys:
		return 1
	case strings.Contains(u, "&CC_"):
		return 2
	case rev:
		return 3
	}
	return 4
}

// matchDriverPackage scans every NVIDIA display INF under dir (recursively) and
// matches it against each present device's hardware IDs, devices in PCI
// address order so the plan is stable.
func matchDriverPackage(devs []GPUBindingDevice, dir string) *GPUDriverPackage {
	cov := &GPUDriverPackage{Dir: dir}
	infs := map[string]map[string]bool{}
	walkErr := filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			if p == dir {
				return err
			}
			cov.Issues = append(cov.Issues, CUDADiscoveryIssue{Path: filepath.ToSlash(p), Error: err.Error()})
			return nil
		}
		if d.IsDir() || !strings.EqualFold(filepath.Ext(p), ".inf") {
			return nil
		}
		b, err := os.ReadFile(p)
		if err != nil {
			cov.Issues = append(cov.Issues, CUDADiscoveryIssue{Path: filepath.ToSlash(p), Error: err.Error()})
			return nil
		}
		text := decodeINF(b)
		if !infIsNVIDIADisplay(text) {
			return nil
		}
		rel, err := filepath.Rel(dir, p)
		if err != nil {
			rel = p
		}
		infs[filepath.ToSlash(rel)] = infHardwareIDs(text)
		return nil
	})
	if walkErr != nil {
		cov.Issues = append(cov.Issues, CUDADiscoveryIssue{Path: filepath.ToSlash(dir), Error: walkErr.Error()})
		return cov
	}
	cov.DisplayINFs = len(infs)
	if cov.DisplayINFs == 0 {
		cov.Issues = append(cov.Issues, CUDADiscoveryIssue{Path: filepath.ToSlash(dir), Error: "no NVIDIA display INF found"})
	}
	names := make([]string, 0, len(infs))
	for name := range infs {
		names = append(names, name)
	}
	sort.Strings(names)

	ordered := append([]GPUBindingDevice(nil), devs...)
	sort.SliceStable(ordered, func(i, j int) bool { return ordered[i].PCIAddress < ordered[j].PCIAddress })
	planned := map[string]bool{}
	for _, dev := range ordered {
		pd := GPUPackageDevice{PCIAddress: dev.PCIAddress, InstanceID: dev.InstanceID}
		best := -1
		for _, name := range names {
			listed := false
			for _, hw := range dev.hardwareIDs {
				if !infs[name][hw] {
					continue
				}
				listed = true
				if r := hardwareIDRank(hw); best < 0 || r < best {
					best, pd.Chosen, pd.MatchedID = r, name, hw
				}
			}
			if listed {
				pd.ListedBy = append(pd.ListedBy, name)
			}
		}
		switch {
		case pd.Chosen == "":
			cov.Uncovered = append(cov.Uncovered, firstNonEmpty(dev.PCIAddress, dev.InstanceID))
		case !planned[pd.Chosen]:
			planned[pd.Chosen] = true
			cov.PnputilCommands = append(cov.PnputilCommands, fmt.Sprintf(`pnputil /add-driver "%s" /install`, filepath.Join(dir, filepath.FromSlash(pd.Chosen))))
		}
		cov.Devices = append(cov.Devices, pd)
	}
	return cov
}
