//go:build darwin

package cluster

import (
	"fmt"
	"os/exec"
	"strings"
)

// QueryRDMAStatus runs `rdma_ctl status` and returns whether RDMA is enabled.
func QueryRDMAStatus() (bool, string, error) {
	out, err := exec.Command("rdma_ctl", "status").CombinedOutput()
	if err != nil {
		outStr := strings.TrimSpace(string(out))
		if strings.Contains(strings.ToLower(outStr), "disabled") {
			return false, "disabled", nil
		}
		return false, "unavailable", fmt.Errorf("rdma_ctl status: %w (%s)", err, outStr)
	}
	capable, status := ParseRDMACtlStatus(string(out))
	return capable, status, nil
}

// TeardownBridge0Conflict constructs and optionally executes commands to dismantle bridge0 hazards:
// deleting members, bringing the bridge down, and disabling the "Thunderbolt Bridge" service.
func TeardownBridge0Conflict(bridgeName string, members []string, dryRun bool) ([]string, error) {
	if bridgeName == "" {
		bridgeName = "bridge0"
	}
	var cmds []string
	for _, m := range members {
		cmds = append(cmds, fmt.Sprintf("ifconfig %s deletem %s", bridgeName, m))
	}
	cmds = append(cmds, fmt.Sprintf("ifconfig %s down", bridgeName))
	cmds = append(cmds, `networksetup -setnetworkserviceenabled "Thunderbolt Bridge" off`)

	if dryRun {
		return cmds, nil
	}

	for _, cmdStr := range cmds {
		var c *exec.Cmd
		if strings.HasPrefix(cmdStr, "networksetup") {
			c = exec.Command("networksetup", "-setnetworkserviceenabled", "Thunderbolt Bridge", "off")
		} else {
			parts := strings.Fields(cmdStr)
			if len(parts) > 0 {
				c = exec.Command(parts[0], parts[1:]...)
			}
		}
		if c != nil {
			if out, err := c.CombinedOutput(); err != nil {
				return cmds, fmt.Errorf("command %q failed: %w (%s)", cmdStr, err, strings.TrimSpace(string(out)))
			}
		}
	}

	return cmds, nil
}

// DiscoverLocalTB5RDMA executes system commands on Darwin to discover local Thunderbolt ports,
// determine link speeds, check RDMA status, detect bridge0 conflicts,
// and assess point-to-point mesh readiness.
func DiscoverLocalTB5RDMA() (*LocalTB5Discovery, error) {
	// 1. Hardware port to BSD name mapping
	hwOut, _ := exec.Command("networksetup", "-listallhardwareports").Output()
	portMap := ParseHardwarePortMap(string(hwOut))

	// 2. Thunderbolt system profiler
	spOut, spErr := exec.Command("system_profiler", "SPThunderboltDataType", "-json").Output()
	var ports []ThunderboltPort
	var parseErr error
	if spErr == nil {
		ports, parseErr = ParseThunderboltSPData(spOut, portMap)
	}

	// 3. RDMA status
	rdmaEnabled, rdmaStatus, _ := QueryRDMAStatus()

	// 4. Bridge0 status
	bridgeOut, _ := exec.Command("ifconfig", "bridge0").CombinedOutput()
	hasConflict, bridgeMembers := DetectBridge0Conflict(string(bridgeOut))

	var tb5Ports []ThunderboltPort
	for _, p := range ports {
		if p.IsTB5 {
			tb5Ports = append(tb5Ports, p)
		}
	}

	disc := &LocalTB5Discovery{
		Ports:             ports,
		TB5Ports:          tb5Ports,
		HasTB5:            len(tb5Ports) > 0,
		RDMAEnabled:       rdmaEnabled,
		RDMAStatus:        rdmaStatus,
		HasBridgeConflict: hasConflict,
		BridgeMembers:     bridgeMembers,
	}

	if hasConflict {
		teardownCmds, _ := TeardownBridge0Conflict("bridge0", bridgeMembers, true)
		disc.TeardownCommands = teardownCmds
		disc.Warnings = append(disc.Warnings, fmt.Sprintf("bridge0 conflict detected (members: %s): broadcast loops will occur in multi-Mac mesh without mitigation", strings.Join(bridgeMembers, ", ")))
	}

	if !rdmaEnabled {
		disc.Warnings = append(disc.Warnings, fmt.Sprintf("RDMA is %s: enable with 'sudo rdma_ctl enable' for zero-copy transfers", rdmaStatus))
	}

	if len(tb5Ports) == 0 {
		disc.Warnings = append(disc.Warnings, "no Thunderbolt 5 (80/120 Gbps) ports detected")
	}

	// Ready when no bridge conflict exists and RDMA is ready
	disc.MeshReady = !hasConflict && rdmaEnabled

	if parseErr != nil {
		return disc, parseErr
	}
	if spErr != nil {
		return disc, spErr
	}
	return disc, nil
}
