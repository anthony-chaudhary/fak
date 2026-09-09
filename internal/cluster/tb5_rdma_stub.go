//go:build !darwin

package cluster

import (
	"errors"
	"fmt"
)

// ErrNonDarwinPlatform indicates that live Thunderbolt 5 RDMA hardware discovery is only available on Darwin.
var ErrNonDarwinPlatform = errors.New("thunderbolt 5 RDMA hardware discovery is only supported on macOS (darwin)")

// QueryRDMAStatus is a stub on !darwin.
func QueryRDMAStatus() (bool, string, error) {
	return false, "unsupported", ErrNonDarwinPlatform
}

// TeardownBridge0Conflict formats teardown commands on !darwin; live execution returns ErrNonDarwinPlatform.
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
	return cmds, ErrNonDarwinPlatform
}

// DiscoverLocalTB5RDMA returns unsupported status on !darwin.
func DiscoverLocalTB5RDMA() (*LocalTB5Discovery, error) {
	return &LocalTB5Discovery{
		Ports:             []ThunderboltPort{},
		TB5Ports:          []ThunderboltPort{},
		HasTB5:            false,
		RDMAEnabled:       false,
		RDMAStatus:        "unsupported",
		HasBridgeConflict: false,
		BridgeMembers:     []string{},
		MeshReady:         false,
		TeardownCommands:  []string{},
		Warnings:          []string{"Thunderbolt 5 RDMA discovery is only supported on macOS (darwin)"},
	}, ErrNonDarwinPlatform
}
