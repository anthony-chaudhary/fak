package cluster

import (
	"testing"
)

func TestThunderbolt5RDMADiscovery_Darwin(t *testing.T) {
	mockSPJSON := []byte(`{
		"SPThunderboltDataType": [
			{
				"_name": "thunderboltusb4_bus_0",
				"device_name_key": "MacBook Pro",
				"domain_uuid_key": "4B3CE3C6-489A-4732-B232-EE373C0601AF",
				"receptacle_1_tag": {
					"current_speed_key": "Up to 80 Gb/s",
					"link_status_key": "0x2",
					"receptacle_id_key": "1",
					"receptacle_status_key": "receptacle_connected"
				},
				"_items": [
					{
						"_name": "connected_device",
						"device_name_key": "Mac Studio Peer",
						"domain_uuid_key": "B3054151-F0F5-4D1F-9AAB-C4C9D4DF0517"
					}
				]
			},
			{
				"_name": "thunderboltusb4_bus_1",
				"device_name_key": "MacBook Pro",
				"domain_uuid_key": "4B3CE3C6-489A-4732-B232-EE373C0601AF",
				"receptacle_1_tag": {
					"current_speed_key": "Up to 120 Gb/s",
					"link_status_key": "0x2",
					"receptacle_id_key": "2",
					"receptacle_status_key": "receptacle_connected"
				},
				"_items": [
					{
						"_name": "connected_device",
						"device_name_key": "Mac Pro Peer",
						"domain_uuid_key": "BE841075-C376-495E-87E8-51786F29C089"
					}
				]
			}
		]
	}`)

	portMap := map[string]string{
		"Thunderbolt 1": "en1",
		"Thunderbolt 2": "en2",
	}

	ports, err := ParseThunderboltSPData(mockSPJSON, portMap)
	if err != nil {
		t.Fatalf("ParseThunderboltSPData failed: %v", err)
	}

	if len(ports) != 2 {
		t.Fatalf("expected 2 ports, got %d", len(ports))
	}

	// Port 1: 80 Gbps TB5
	p1 := ports[0]
	if p1.PortName != "Thunderbolt 1" {
		t.Errorf("p1.PortName = %q, want 'Thunderbolt 1'", p1.PortName)
	}
	if p1.BSDName != "en1" {
		t.Errorf("p1.BSDName = %q, want 'en1'", p1.BSDName)
	}
	if p1.RDMAInterface != "rdma_en1" {
		t.Errorf("p1.RDMAInterface = %q, want 'rdma_en1'", p1.RDMAInterface)
	}
	if p1.SpeedGbps != 80 {
		t.Errorf("p1.SpeedGbps = %d, want 80", p1.SpeedGbps)
	}
	if !p1.IsTB5 {
		t.Errorf("p1.IsTB5 = false, want true")
	}
	if !p1.Connected {
		t.Errorf("p1.Connected = false, want true")
	}
	if p1.DomainUUID != "4B3CE3C6-489A-4732-B232-EE373C0601AF" {
		t.Errorf("p1.DomainUUID = %q, want local domain", p1.DomainUUID)
	}
	if p1.RemoteDomainUUID != "B3054151-F0F5-4D1F-9AAB-C4C9D4DF0517" {
		t.Errorf("p1.RemoteDomainUUID = %q, want sink domain", p1.RemoteDomainUUID)
	}

	// Port 2: 120 Gbps TB5
	p2 := ports[1]
	if p2.PortName != "Thunderbolt 2" {
		t.Errorf("p2.PortName = %q, want 'Thunderbolt 2'", p2.PortName)
	}
	if p2.BSDName != "en2" {
		t.Errorf("p2.BSDName = %q, want 'en2'", p2.BSDName)
	}
	if p2.RDMAInterface != "rdma_en2" {
		t.Errorf("p2.RDMAInterface = %q, want 'rdma_en2'", p2.RDMAInterface)
	}
	if p2.SpeedGbps != 120 {
		t.Errorf("p2.SpeedGbps = %d, want 120", p2.SpeedGbps)
	}
	if !p2.IsTB5 {
		t.Errorf("p2.IsTB5 = false, want true")
	}
	if !p2.Connected {
		t.Errorf("p2.Connected = false, want true")
	}
	if p2.RemoteDomainUUID != "BE841075-C376-495E-87E8-51786F29C089" {
		t.Errorf("p2.RemoteDomainUUID = %q, want sink domain", p2.RemoteDomainUUID)
	}
}

func TestHardwarePortMapParsing(t *testing.T) {
	output := `
Hardware Port: Ethernet Adapter (en4)
Device: en4
Ethernet Address: f6:f6:d2:95:38:17

Hardware Port: Thunderbolt Bridge
Device: bridge0
Ethernet Address: 36:f8:fd:d4:09:00

Hardware Port: Thunderbolt 1
Device: en1
Ethernet Address: 36:f8:fd:d4:09:00

Hardware Port: Thunderbolt 2
Device: en2
Ethernet Address: 36:f8:fd:d4:09:04
`
	m := ParseHardwarePortMap(output)
	if m["Thunderbolt 1"] != "en1" {
		t.Errorf("Thunderbolt 1 = %q, want 'en1'", m["Thunderbolt 1"])
	}
	if m["1"] != "en1" {
		t.Errorf("1 = %q, want 'en1'", m["1"])
	}
	if m["Thunderbolt 2"] != "en2" {
		t.Errorf("Thunderbolt 2 = %q, want 'en2'", m["Thunderbolt 2"])
	}
	if m["Thunderbolt Bridge"] != "bridge0" {
		t.Errorf("Thunderbolt Bridge = %q, want 'bridge0'", m["Thunderbolt Bridge"])
	}
}

func TestBridge0ConflictAndMitigation(t *testing.T) {
	mockIfconfig := `bridge0: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
	options=63<RXCSUM,TXCSUM,TSO4,TSO6>
	ether 36:f8:fd:d4:09:00
	Configuration:
		id 0:0:0:0:0:0 priority 0 hellotime 0 fwddelay 0
	member: en1 flags=3<LEARNING,DISCOVER>
	        ifmaxaddr 0 port 10 priority 0 path cost 0
	member: en2 flags=3<LEARNING,DISCOVER>
	        ifmaxaddr 0 port 11 priority 0 path cost 0
	status: active
`
	hasConflict, members := DetectBridge0Conflict(mockIfconfig)
	if !hasConflict {
		t.Fatalf("expected hasConflict = true, got false")
	}
	if len(members) != 2 || members[0] != "en1" || members[1] != "en2" {
		t.Fatalf("unexpected members: %v, want [en1, en2]", members)
	}

	cmds, err := TeardownBridge0Conflict("bridge0", members, true)
	if err != nil {
		t.Fatalf("TeardownBridge0Conflict dryRun error: %v", err)
	}

	wantCmds := []string{
		"ifconfig bridge0 deletem en1",
		"ifconfig bridge0 deletem en2",
		"ifconfig bridge0 down",
		`networksetup -setnetworkserviceenabled "Thunderbolt Bridge" off`,
	}

	if len(cmds) != len(wantCmds) {
		t.Fatalf("got %d cmds, want %d: %v", len(cmds), len(wantCmds), cmds)
	}
	for i, c := range wantCmds {
		if cmds[i] != c {
			t.Errorf("cmd[%d] = %q, want %q", i, cmds[i], c)
		}
	}

	// Clean bridge (no members)
	cleanIfconfig := `bridge0: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
	status: inactive
`
	cleanConflict, cleanMembers := DetectBridge0Conflict(cleanIfconfig)
	if cleanConflict {
		t.Errorf("expected cleanConflict = false, got true")
	}
	if len(cleanMembers) != 0 {
		t.Errorf("expected empty members, got %v", cleanMembers)
	}
}

func TestMeshTopologyAllocation(t *testing.T) {
	// 1. 2-Mac topology (1 TB5 link @ 80 Gbps)
	peers2 := []MeshPeer{
		{
			ID:         "mac1",
			DomainUUID: "uuid-1",
			Ports: []ThunderboltPort{
				{
					BSDName:          "en1",
					PortName:         "Thunderbolt 1",
					SpeedGbps:        80,
					LinkSpeed:        "Up to 80 Gb/s",
					Connected:        true,
					IsTB5:            true,
					DomainUUID:       "uuid-1",
					RemoteDomainUUID: "uuid-2",
				},
			},
		},
		{
			ID:         "mac2",
			DomainUUID: "uuid-2",
			Ports: []ThunderboltPort{
				{
					BSDName:          "en1",
					PortName:         "Thunderbolt 1",
					SpeedGbps:        80,
					LinkSpeed:        "Up to 80 Gb/s",
					Connected:        true,
					IsTB5:            true,
					DomainUUID:       "uuid-2",
					RemoteDomainUUID: "uuid-1",
				},
			},
		},
	}

	topo2, err := AllocateMeshSubnets(peers2, "10.55.0.0/16")
	if err != nil {
		t.Fatalf("2-Mac topology allocation failed: %v", err)
	}
	if len(topo2.Links) != 1 {
		t.Fatalf("2-Mac expected 1 link, got %d", len(topo2.Links))
	}
	l1 := topo2.Links[0]
	if l1.SubnetCIDR != "10.55.1.0/30" {
		t.Errorf("2-Mac SubnetCIDR = %q, want '10.55.1.0/30'", l1.SubnetCIDR)
	}
	if l1.PeerAIP != "10.55.1.1" || l1.PeerBIP != "10.55.1.2" {
		t.Errorf("2-Mac IPs = (%s, %s), want (10.55.1.1, 10.55.1.2)", l1.PeerAIP, l1.PeerBIP)
	}
	if l1.Netmask != "255.255.255.252" {
		t.Errorf("2-Mac Netmask = %q, want '255.255.255.252'", l1.Netmask)
	}
	if topo2.TotalBandwidthGbps != 80 {
		t.Errorf("2-Mac TotalBandwidthGbps = %d, want 80", topo2.TotalBandwidthGbps)
	}

	// 2. 3-Mac topology (triangle/mesh, links @ 80, 80, 120 Gbps)
	peers3 := []MeshPeer{
		{
			ID:         "mac1",
			DomainUUID: "uuid-1",
			Ports: []ThunderboltPort{
				{
					BSDName:          "en1",
					SpeedGbps:        80,
					Connected:        true,
					DomainUUID:       "uuid-1",
					RemoteDomainUUID: "uuid-2",
				},
				{
					BSDName:          "en2",
					SpeedGbps:        120,
					Connected:        true,
					DomainUUID:       "uuid-1",
					RemoteDomainUUID: "uuid-3",
				},
			},
		},
		{
			ID:         "mac2",
			DomainUUID: "uuid-2",
			Ports: []ThunderboltPort{
				{
					BSDName:          "en1",
					SpeedGbps:        80,
					Connected:        true,
					DomainUUID:       "uuid-2",
					RemoteDomainUUID: "uuid-1",
				},
				{
					BSDName:          "en2",
					SpeedGbps:        80,
					Connected:        true,
					DomainUUID:       "uuid-2",
					RemoteDomainUUID: "uuid-3",
				},
			},
		},
		{
			ID:         "mac3",
			DomainUUID: "uuid-3",
			Ports: []ThunderboltPort{
				{
					BSDName:          "en1",
					SpeedGbps:        120,
					Connected:        true,
					DomainUUID:       "uuid-3",
					RemoteDomainUUID: "uuid-1",
				},
				{
					BSDName:          "en2",
					SpeedGbps:        80,
					Connected:        true,
					DomainUUID:       "uuid-3",
					RemoteDomainUUID: "uuid-2",
				},
			},
		},
	}

	topo3, err := AllocateMeshSubnets(peers3, "10.55.0.0/16")
	if err != nil {
		t.Fatalf("3-Mac topology allocation failed: %v", err)
	}
	if len(topo3.Links) != 3 {
		t.Fatalf("3-Mac expected 3 links, got %d", len(topo3.Links))
	}
	wantSubnets3 := []string{"10.55.1.0/30", "10.55.2.0/30", "10.55.3.0/30"}
	for i, s := range wantSubnets3 {
		if topo3.Links[i].SubnetCIDR != s {
			t.Errorf("3-Mac link[%d] subnet = %q, want %q", i, topo3.Links[i].SubnetCIDR, s)
		}
	}
	// Total bandwidth: 80 + 120 + 80 = 280 Gbps
	if topo3.TotalBandwidthGbps != 280 {
		t.Errorf("3-Mac TotalBandwidthGbps = %d, want 280", topo3.TotalBandwidthGbps)
	}

	// 3. 4-Mac topology (ring: 4 links @ 80 Gbps each)
	peers4 := []MeshPeer{
		{
			ID:         "mac1",
			DomainUUID: "uuid-1",
			Ports: []ThunderboltPort{
				{BSDName: "en1", SpeedGbps: 80, Connected: true, RemoteDomainUUID: "uuid-2"},
				{BSDName: "en2", SpeedGbps: 80, Connected: true, RemoteDomainUUID: "uuid-4"},
			},
		},
		{
			ID:         "mac2",
			DomainUUID: "uuid-2",
			Ports: []ThunderboltPort{
				{BSDName: "en1", SpeedGbps: 80, Connected: true, RemoteDomainUUID: "uuid-1"},
				{BSDName: "en2", SpeedGbps: 80, Connected: true, RemoteDomainUUID: "uuid-3"},
			},
		},
		{
			ID:         "mac3",
			DomainUUID: "uuid-3",
			Ports: []ThunderboltPort{
				{BSDName: "en1", SpeedGbps: 80, Connected: true, RemoteDomainUUID: "uuid-2"},
				{BSDName: "en2", SpeedGbps: 80, Connected: true, RemoteDomainUUID: "uuid-4"},
			},
		},
		{
			ID:         "mac4",
			DomainUUID: "uuid-4",
			Ports: []ThunderboltPort{
				{BSDName: "en1", SpeedGbps: 80, Connected: true, RemoteDomainUUID: "uuid-3"},
				{BSDName: "en2", SpeedGbps: 80, Connected: true, RemoteDomainUUID: "uuid-1"},
			},
		},
	}

	topo4, err := AllocateMeshSubnets(peers4, "10.55.0.0/16")
	if err != nil {
		t.Fatalf("4-Mac topology allocation failed: %v", err)
	}
	if len(topo4.Links) != 4 {
		t.Fatalf("4-Mac expected 4 links, got %d", len(topo4.Links))
	}
	if topo4.TotalBandwidthGbps != 320 {
		t.Errorf("4-Mac TotalBandwidthGbps = %d, want 320", topo4.TotalBandwidthGbps)
	}

	// 4. Bridge conflict verification
	peersWithBridge := []MeshPeer{
		{
			ID:            "mac1",
			DomainUUID:    "uuid-1",
			BridgeMembers: []string{"en1"},
			Ports: []ThunderboltPort{
				{BSDName: "en1", SpeedGbps: 80, Connected: true, RemoteDomainUUID: "uuid-2"},
			},
		},
		{
			ID:         "mac2",
			DomainUUID: "uuid-2",
			Ports: []ThunderboltPort{
				{BSDName: "en1", SpeedGbps: 80, Connected: true, RemoteDomainUUID: "uuid-1"},
			},
		},
	}
	topoConflict, err := AllocateMeshSubnets(peersWithBridge, "10.55.0.0/16")
	if err != nil {
		t.Fatalf("topology allocation failed: %v", err)
	}
	if !topoConflict.HasBridgeConflict {
		t.Errorf("expected HasBridgeConflict = true")
	}
	if len(topoConflict.BridgeConflictMembers) == 0 || topoConflict.BridgeConflictMembers[0] != "en1" {
		t.Errorf("expected BridgeConflictMembers to contain 'en1', got %v", topoConflict.BridgeConflictMembers)
	}
}

func TestRDMACtlStatusParsing(t *testing.T) {
	tests := []struct {
		input       string
		wantCapable bool
		wantStatus  string
	}{
		{"enabled", true, "enabled"},
		{"ENABLED\n", true, "enabled"},
		{"disabled", false, "disabled"},
		{"  DISABLED \n", false, "disabled"},
		{"unavailable", false, "unavailable"},
		{"", false, "unavailable"},
		{"error: device not found", false, "error: device not found"},
	}

	for _, tt := range tests {
		capable, status := ParseRDMACtlStatus(tt.input)
		if capable != tt.wantCapable {
			t.Errorf("ParseRDMACtlStatus(%q) capable = %t, want %t", tt.input, capable, tt.wantCapable)
		}
		if status != tt.wantStatus {
			t.Errorf("ParseRDMACtlStatus(%q) status = %q, want %q", tt.input, status, tt.wantStatus)
		}
	}
}
