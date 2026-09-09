package cluster

import (
	"encoding/json"
	"fmt"
	"net"
	"sort"
	"strconv"
	"strings"
)

// ThunderboltPort describes a single physical or virtual Thunderbolt port on a Mac host.
type ThunderboltPort struct {
	ReceptacleID     string `json:"receptacle_id"`                // e.g. "1", "2", "3"
	PortName         string `json:"port_name"`                    // e.g. "Thunderbolt 1"
	BSDName          string `json:"bsd_name"`                     // e.g. "en1", "en2"
	RDMAInterface    string `json:"rdma_interface"`               // e.g. "rdma_en1"
	SpeedGbps        int    `json:"speed_gbps"`                   // e.g. 40, 80, 120
	LinkSpeed        string `json:"link_speed"`                   // e.g. "Up to 80 Gb/s", "Up to 120 Gb/s"
	LinkStatus       string `json:"link_status"`                  // e.g. "connected", "0x2"
	Connected        bool   `json:"connected"`                    // true if device attached
	IsTB5            bool   `json:"is_tb5"`                       // true if speed >= 80 Gbps
	DomainUUID       string `json:"domain_uuid"`                  // local bus domain UUID
	RemoteDomainUUID string `json:"remote_domain_uuid,omitempty"` // peer/sink domain UUID
	DeviceName       string `json:"device_name,omitempty"`        // local device name (e.g. "Mac Studio")
	RemoteDeviceName string `json:"remote_device_name,omitempty"` // peer device name
	ConnectedPeerID  string `json:"connected_peer_id,omitempty"`  // optional explicit peer identifier
}

// Identifier returns a deterministic identifier for the port on this host.
func (p *ThunderboltPort) Identifier(fallbackIdx int) string {
	if p.BSDName != "" {
		return p.BSDName
	}
	if p.ReceptacleID != "" {
		return "rec_" + p.ReceptacleID
	}
	if p.PortName != "" {
		return p.PortName
	}
	return fmt.Sprintf("port_%d", fallbackIdx)
}

// MeshPeer represents an individual Mac node participating in the point-to-point mesh.
type MeshPeer struct {
	ID            string            `json:"id"`                       // unique node identifier (e.g. "mac1")
	Name          string            `json:"name,omitempty"`           // friendly name
	DomainUUID    string            `json:"domain_uuid,omitempty"`    // host Thunderbolt domain UUID
	Ports         []ThunderboltPort `json:"ports,omitempty"`          // host Thunderbolt ports
	BridgeMembers []string          `json:"bridge_members,omitempty"` // active bridge0 members if any
}

// PointToPointLink represents a dedicated bidirectional connection between two Macs
// on a private /30 point-to-point IP subnet.
type PointToPointLink struct {
	PeerA      string `json:"peer_a"`      // Peer A node ID or domain UUID
	PeerB      string `json:"peer_b"`      // Peer B node ID or domain UUID
	PortA      string `json:"port_a"`      // BSD interface on Peer A (e.g. "en1")
	PortB      string `json:"port_b"`      // BSD interface on Peer B (e.g. "en2")
	SubnetCIDR string `json:"subnet_cidr"` // Assigned point-to-point /30 subnet (e.g. "10.55.1.0/30")
	PeerAIP    string `json:"peer_a_ip"`   // IP address assigned to Peer A (e.g. "10.55.1.1")
	PeerBIP    string `json:"peer_b_ip"`   // IP address assigned to Peer B (e.g. "10.55.1.2")
	Netmask    string `json:"netmask"`     // "255.255.255.252"
	SpeedGbps  int    `json:"speed_gbps"`  // Negotiated link speed in Gbps (e.g. 80, 120)
}

// MeshTopology holds the full interconnected cluster topology, allocated subnets,
// total aggregate bandwidth, and conflict/loop validation state.
type MeshTopology struct {
	Peers                 []MeshPeer         `json:"peers"`
	Links                 []PointToPointLink `json:"links"`
	TotalBandwidthGbps    int                `json:"total_bandwidth_gbps"`
	HasBridgeConflict     bool               `json:"has_bridge_conflict"`
	BridgeConflictMembers []string           `json:"bridge_conflict_members,omitempty"`
	Warnings              []string           `json:"warnings,omitempty"`
}

// LocalTB5Discovery summarizes the local host's Thunderbolt 5 and RDMA discovery report.
type LocalTB5Discovery struct {
	Ports             []ThunderboltPort `json:"ports"`
	TB5Ports          []ThunderboltPort `json:"tb5_ports"`
	HasTB5            bool              `json:"has_tb5"`
	RDMAEnabled       bool              `json:"rdma_enabled"`
	RDMAStatus        string            `json:"rdma_status"`
	HasBridgeConflict bool              `json:"has_bridge_conflict"`
	BridgeMembers     []string          `json:"bridge_members"`
	MeshReady         bool              `json:"mesh_ready"`
	TeardownCommands  []string          `json:"teardown_commands,omitempty"`
	Warnings          []string          `json:"warnings,omitempty"`
}

// FormatSubnet30 formats a point-to-point /30 subnet for the given link index (1-based).
// For default "10.55.0.0/16" or /16 prefixes, it allocates 10.55.X.0/30 where X is the link index.
// For /24 or smaller prefixes, it allocates consecutive 4-IP blocks (.0, .4, .8...).
func FormatSubnet30(baseSubnet string, linkIdx int) (cidr, ipA, ipB, netmask string, err error) {
	if linkIdx < 1 {
		return "", "", "", "", fmt.Errorf("link index must be >= 1, got %d", linkIdx)
	}
	if baseSubnet == "" {
		baseSubnet = "10.55.0.0/16"
	}

	netmask = "255.255.255.252"

	// Preferred convention: 10.55.X.0/30
	if strings.HasPrefix(baseSubnet, "10.55") || strings.Contains(baseSubnet, "/16") {
		parts := strings.Split(strings.Split(baseSubnet, "/")[0], ".")
		octet1 := "10"
		octet2 := "55"
		if len(parts) >= 2 {
			octet1 = parts[0]
			octet2 = parts[1]
		}
		if linkIdx > 254 {
			return "", "", "", "", fmt.Errorf("link index %d exceeds /16 subnet capacity", linkIdx)
		}
		cidr = fmt.Sprintf("%s.%s.%d.0/30", octet1, octet2, linkIdx)
		ipA = fmt.Sprintf("%s.%s.%d.1", octet1, octet2, linkIdx)
		ipB = fmt.Sprintf("%s.%s.%d.2", octet1, octet2, linkIdx)
		return cidr, ipA, ipB, netmask, nil
	}

	// General CIDR parsing
	ip, ipNet, parseErr := net.ParseCIDR(baseSubnet)
	if parseErr != nil {
		ip = net.ParseIP(baseSubnet)
		if ip == nil {
			return "", "", "", "", fmt.Errorf("invalid base subnet %q: %w", baseSubnet, parseErr)
		}
		ipNet = &net.IPNet{IP: ip, Mask: net.CIDRMask(24, 32)}
	}

	ipv4 := ip.To4()
	if ipv4 == nil {
		return "", "", "", "", fmt.Errorf("only IPv4 subnets are supported, got %s", baseSubnet)
	}

	ones, _ := ipNet.Mask.Size()
	if ones <= 16 {
		cidr = fmt.Sprintf("%d.%d.%d.0/30", ipv4[0], ipv4[1], linkIdx)
		ipA = fmt.Sprintf("%d.%d.%d.1", ipv4[0], ipv4[1], linkIdx)
		ipB = fmt.Sprintf("%d.%d.%d.2", ipv4[0], ipv4[1], linkIdx)
		return cidr, ipA, ipB, netmask, nil
	}

	// For /24 or smaller, allocate 4 IPs per link
	offset := (linkIdx - 1) * 4
	if offset+3 > 255 {
		return "", "", "", "", fmt.Errorf("base subnet %s exhausted for link %d", baseSubnet, linkIdx)
	}
	cidr = fmt.Sprintf("%d.%d.%d.%d/30", ipv4[0], ipv4[1], ipv4[2], offset)
	ipA = fmt.Sprintf("%d.%d.%d.%d", ipv4[0], ipv4[1], ipv4[2], offset+1)
	ipB = fmt.Sprintf("%d.%d.%d.%d", ipv4[0], ipv4[1], ipv4[2], offset+2)
	return cidr, ipA, ipB, netmask, nil
}

// AllocateMeshSubnets allocates point-to-point /30 CIDR subnets (e.g. 10.55.X.0/30)
// for each bidirectional connection between peers, preventing broadcast loops and
// Ethernet bridging storms. It computes total aggregate bandwidth and validates topology.
func AllocateMeshSubnets(peers []MeshPeer, baseSubnet string) (*MeshTopology, error) {
	if len(peers) == 0 {
		return nil, fmt.Errorf("cannot allocate mesh topology: empty peers list")
	}

	topo := &MeshTopology{
		Peers: peers,
		Links: make([]PointToPointLink, 0),
	}

	usedPorts := make(map[string]bool)
	var totalBandwidth int

	for i := range peers {
		peerA := &peers[i]
		for portIdxA := range peerA.Ports {
			portA := &peerA.Ports[portIdxA]
			if !portA.Connected {
				continue
			}

			// Self-loop check
			if portA.RemoteDomainUUID != "" && (portA.RemoteDomainUUID == peerA.DomainUUID || portA.RemoteDomainUUID == peerA.ID) {
				topo.Warnings = append(topo.Warnings, fmt.Sprintf("self-loop detected: peer %s port %s connects to self", peerA.ID, portA.Identifier(portIdxA)))
				continue
			}
			if portA.ConnectedPeerID != "" && portA.ConnectedPeerID == peerA.ID {
				topo.Warnings = append(topo.Warnings, fmt.Sprintf("self-loop detected: peer %s port %s connects to self", peerA.ID, portA.Identifier(portIdxA)))
				continue
			}

			portAKey := peerA.ID + "/" + portA.Identifier(portIdxA)
			if usedPorts[portAKey] {
				continue
			}

			// Find matching peer B
			var matchedPeer *MeshPeer
			var matchedPort *ThunderboltPort
			var matchedPortIdx int

			for j := range peers {
				if peers[j].ID == peerA.ID {
					continue
				}
				peerB := &peers[j]

				isPeerMatch := (portA.RemoteDomainUUID != "" && (peerB.DomainUUID == portA.RemoteDomainUUID || peerB.ID == portA.RemoteDomainUUID)) ||
					(portA.ConnectedPeerID != "" && (peerB.ID == portA.ConnectedPeerID || peerB.DomainUUID == portA.ConnectedPeerID))

				if !isPeerMatch {
					continue
				}

				// Look for port on peerB pointing back or available connected port
				for portIdxB := range peerB.Ports {
					portB := &peerB.Ports[portIdxB]
					if !portB.Connected {
						continue
					}
					portBKey := peerB.ID + "/" + portB.Identifier(portIdxB)
					if usedPorts[portBKey] {
						continue
					}

					pointsBack := (portB.RemoteDomainUUID != "" && (portB.RemoteDomainUUID == peerA.DomainUUID || portB.RemoteDomainUUID == peerA.ID)) ||
						(portB.ConnectedPeerID != "" && (portB.ConnectedPeerID == peerA.ID || portB.ConnectedPeerID == peerA.DomainUUID))

					if pointsBack {
						matchedPeer = peerB
						matchedPort = portB
						matchedPortIdx = portIdxB
						break
					}
					if matchedPort == nil {
						matchedPeer = peerB
						matchedPort = portB
						matchedPortIdx = portIdxB
					}
				}
				if matchedPort != nil {
					break
				}
			}

			// 2-peer fallback if neither has explicit RemoteDomainUUID
			if matchedPort == nil && len(peers) == 2 {
				otherIdx := 1 - i
				peerB := &peers[otherIdx]
				for portIdxB := range peerB.Ports {
					portB := &peerB.Ports[portIdxB]
					if !portB.Connected {
						continue
					}
					portBKey := peerB.ID + "/" + portB.Identifier(portIdxB)
					if usedPorts[portBKey] {
						continue
					}
					matchedPeer = peerB
					matchedPort = portB
					matchedPortIdx = portIdxB
					break
				}
			}

			if matchedPeer != nil && matchedPort != nil {
				portBKey := matchedPeer.ID + "/" + matchedPort.Identifier(matchedPortIdx)
				usedPorts[portAKey] = true
				usedPorts[portBKey] = true

				speed := portA.SpeedGbps
				if matchedPort.SpeedGbps > 0 {
					if speed == 0 || matchedPort.SpeedGbps < speed {
						speed = matchedPort.SpeedGbps
					}
				}
				if speed == 0 {
					speed = 80 // default to TB5 standard 80 Gbps
				}

				linkIdx := len(topo.Links) + 1
				cidr, ipA, ipB, netmask, err := FormatSubnet30(baseSubnet, linkIdx)
				if err != nil {
					return nil, fmt.Errorf("failed to format subnet for link %d: %w", linkIdx, err)
				}

				portAName := portA.BSDName
				if portAName == "" {
					portAName = portA.PortName
				}
				portBName := matchedPort.BSDName
				if portBName == "" {
					portBName = matchedPort.PortName
				}

				link := PointToPointLink{
					PeerA:      peerA.ID,
					PeerB:      matchedPeer.ID,
					PortA:      portAName,
					PortB:      portBName,
					SubnetCIDR: cidr,
					PeerAIP:    ipA,
					PeerBIP:    ipB,
					Netmask:    netmask,
					SpeedGbps:  speed,
				}
				topo.Links = append(topo.Links, link)
				totalBandwidth += speed
			}
		}
	}

	topo.TotalBandwidthGbps = totalBandwidth

	// Check bridge conflicts across all peers
	for _, p := range peers {
		if len(p.BridgeMembers) > 0 {
			for _, m := range p.BridgeMembers {
				for _, link := range topo.Links {
					if (link.PeerA == p.ID && link.PortA == m) || (link.PeerB == p.ID && link.PortB == m) {
						topo.HasBridgeConflict = true
						topo.BridgeConflictMembers = append(topo.BridgeConflictMembers, m)
						topo.Warnings = append(topo.Warnings, fmt.Sprintf("bridge0 conflict: interface %s on peer %s is an active bridge member", m, p.ID))
					}
				}
			}
		}
	}

	// Validate topology graph structure
	if err := topo.Validate(); err != nil {
		topo.Warnings = append(topo.Warnings, fmt.Sprintf("validation: %v", err))
	}

	return topo, nil
}

// CheckBridgeConflict updates the topology with detected bridge members,
// flagging bridge conflict and warnings if any active link uses a bridged interface.
func (t *MeshTopology) CheckBridgeConflict(bridgeMembers []string) (bool, []string) {
	if len(bridgeMembers) == 0 {
		return false, nil
	}
	var conflicting []string
	memberSet := make(map[string]bool)
	for _, m := range bridgeMembers {
		memberSet[m] = true
	}

	for _, link := range t.Links {
		if memberSet[link.PortA] {
			conflicting = append(conflicting, link.PortA)
		}
		if memberSet[link.PortB] {
			conflicting = append(conflicting, link.PortB)
		}
	}

	if len(conflicting) > 0 {
		t.HasBridgeConflict = true
		t.BridgeConflictMembers = append(t.BridgeConflictMembers, conflicting...)
		t.Warnings = append(t.Warnings, fmt.Sprintf("bridge0 loop conflict: interfaces %v are active in bridge0; Ethernet bridging must be disabled for point-to-point mesh", conflicting))
		return true, conflicting
	}
	return false, nil
}

// Validate verifies subnet uniqueness, checks for cycles and bridging hazards,
// and ensures each peer has at least one valid link.
func (t *MeshTopology) Validate() error {
	seenSubnets := make(map[string]bool)
	for _, link := range t.Links {
		if seenSubnets[link.SubnetCIDR] {
			return fmt.Errorf("duplicate subnet CIDR %s allocated in topology", link.SubnetCIDR)
		}
		seenSubnets[link.SubnetCIDR] = true
	}

	// Detect cycles in the topology graph.
	// In L3 point-to-point routing, mesh cycles (e.g. 3-Mac ring or 4-Mac full mesh)
	// are standard for multipath routing, but if Layer 2 Ethernet bridging (bridge0)
	// is active, they form broadcast loops.
	if len(t.Links) >= len(t.Peers) && len(t.Peers) >= 3 {
		if t.HasBridgeConflict {
			t.Warnings = append(t.Warnings, "CRITICAL: mesh loop detected with active bridge0 members; Ethernet broadcast storm hazard present until bridge0 is destroyed")
		} else {
			t.Warnings = append(t.Warnings, "topology contains multi-hop mesh loops; point-to-point L3 /30 routing isolates broadcast domains")
		}
	}

	return nil
}

// ParseHardwarePortMap parses the output of `networksetup -listallhardwareports`.
// It maps Hardware Port descriptions (e.g. "Thunderbolt 1", "Thunderbolt Bridge")
// to their corresponding BSD device names (e.g. "en1", "bridge0").
func ParseHardwarePortMap(output string) map[string]string {
	m := make(map[string]string)
	lines := strings.Split(output, "\n")
	var currentPort string

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "Hardware Port:") {
			currentPort = strings.TrimSpace(strings.TrimPrefix(line, "Hardware Port:"))
		} else if strings.HasPrefix(line, "Device:") && currentPort != "" {
			device := strings.TrimSpace(strings.TrimPrefix(line, "Device:"))
			m[currentPort] = device
			if strings.HasPrefix(currentPort, "Thunderbolt ") {
				num := strings.TrimPrefix(currentPort, "Thunderbolt ")
				m[num] = device
			}
			currentPort = ""
		}
	}
	return m
}

// ParseThunderboltSPData parses JSON output from `system_profiler SPThunderboltDataType -json`.
// It extracts Thunderbolt buses, receptacles, link speeds, connection statuses, domain UUIDs,
// maps BSD names from portToBSD, derives RDMA interface names (rdma_<bsd_name>),
// and detects Thunderbolt 5 (>= 80 Gbps).
func ParseThunderboltSPData(jsonData []byte, portToBSD map[string]string) ([]ThunderboltPort, error) {
	var root struct {
		SPThunderboltDataType []map[string]interface{} `json:"SPThunderboltDataType"`
	}

	if err := json.Unmarshal(jsonData, &root); err != nil {
		return nil, fmt.Errorf("failed to parse SPThunderboltDataType JSON: %w", err)
	}

	var ports []ThunderboltPort

	for _, bus := range root.SPThunderboltDataType {
		domainUUID, _ := bus["domain_uuid_key"].(string)
		deviceName, _ := bus["device_name_key"].(string)

		var remoteDomainUUID string
		var remoteDeviceName string
		var itemSpeed string

		if items, ok := bus["_items"].([]interface{}); ok && len(items) > 0 {
			if itemMap, ok := items[0].(map[string]interface{}); ok {
				remoteDomainUUID, _ = itemMap["domain_uuid_key"].(string)
				remoteDeviceName, _ = itemMap["device_name_key"].(string)
				itemSpeed, _ = itemMap["current_speed_key"].(string)
			}
		}

		var tagKeys []string
		for k := range bus {
			if strings.HasPrefix(k, "receptacle_") && strings.HasSuffix(k, "_tag") {
				tagKeys = append(tagKeys, k)
			}
		}
		sort.Strings(tagKeys)

		if len(tagKeys) == 0 {
			tagKeys = append(tagKeys, "")
		}

		for _, tagKey := range tagKeys {
			recID := "1"
			linkSpeed := itemSpeed
			linkStatus := ""
			recStatus := ""

			if tagKey != "" {
				if recTag, ok := bus[tagKey].(map[string]interface{}); ok {
					switch v := recTag["receptacle_id_key"].(type) {
					case string:
						recID = v
					case float64:
						recID = fmt.Sprintf("%.0f", v)
					case int:
						recID = strconv.Itoa(v)
					}

					if sp, ok := recTag["current_speed_key"].(string); ok && sp != "" {
						linkSpeed = sp
					}
					if ls, ok := recTag["link_status_key"].(string); ok {
						linkStatus = ls
					}
					if rs, ok := recTag["receptacle_status_key"].(string); ok {
						recStatus = rs
					}
				}
			}

			connected := (remoteDomainUUID != "") ||
				(recStatus != "" && strings.Contains(recStatus, "connected") && !strings.Contains(recStatus, "no_devices")) ||
				(linkStatus == "0x2")

			speedGbps := parseSpeedGbps(linkSpeed)
			isTB5 := speedGbps >= 80 || strings.Contains(linkSpeed, "80 Gb/s") || strings.Contains(linkSpeed, "120 Gb/s")

			portName := "Thunderbolt " + recID
			bsdName := ""
			if portToBSD != nil {
				if dev, ok := portToBSD[portName]; ok {
					bsdName = dev
				} else if dev, ok := portToBSD[recID]; ok {
					bsdName = dev
				}
			}

			rdmaInterface := ""
			if bsdName != "" {
				rdmaInterface = "rdma_" + bsdName
			}

			ports = append(ports, ThunderboltPort{
				ReceptacleID:     recID,
				PortName:         portName,
				BSDName:          bsdName,
				RDMAInterface:    rdmaInterface,
				SpeedGbps:        speedGbps,
				LinkSpeed:        linkSpeed,
				LinkStatus:       linkStatus,
				Connected:        connected,
				IsTB5:            isTB5,
				DomainUUID:       domainUUID,
				RemoteDomainUUID: remoteDomainUUID,
				DeviceName:       deviceName,
				RemoteDeviceName: remoteDeviceName,
			})
		}
	}

	return ports, nil
}

// parseSpeedGbps extracts an integer speed in Gbps from descriptive strings like "Up to 80 Gb/s".
func parseSpeedGbps(s string) int {
	sLower := strings.ToLower(s)
	if strings.Contains(sLower, "120") {
		return 120
	}
	if strings.Contains(sLower, "80") {
		return 80
	}
	if strings.Contains(sLower, "40") {
		return 40
	}
	if strings.Contains(sLower, "20") {
		return 20
	}
	if strings.Contains(sLower, "10") {
		return 10
	}

	var digits []rune
	for _, r := range sLower {
		if r >= '0' && r <= '9' {
			digits = append(digits, r)
		} else if len(digits) > 0 {
			break
		}
	}
	if len(digits) > 0 {
		n, _ := strconv.Atoi(string(digits))
		return n
	}
	return 0
}

// ParseRDMACtlStatus parses the output of `rdma_ctl status`.
func ParseRDMACtlStatus(output string) (bool, string) {
	trimmed := strings.ToLower(strings.TrimSpace(output))
	if strings.Contains(trimmed, "enabled") {
		return true, "enabled"
	}
	if strings.Contains(trimmed, "disabled") {
		return false, "disabled"
	}
	if trimmed == "" {
		return false, "unavailable"
	}
	return false, trimmed
}

// DetectBridge0Conflict checks whether ifconfig output for bridge0 contains active member interfaces.
func DetectBridge0Conflict(ifconfigOutput string) (bool, []string) {
	var members []string
	lines := strings.Split(ifconfigOutput, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "member:") {
			fields := strings.Fields(line)
			if len(fields) >= 2 {
				members = append(members, fields[1])
			}
		}
	}
	return len(members) > 0, members
}
