package gateway

import (
	"fmt"
	"sort"
	"strings"
)

// UCXTransportType represents the chosen transport for disaggregated KV transfer.
type UCXTransportType string

const (
	UCXTransportInfiniBand UCXTransportType = "ib"
	UCXTransportCUDAIPC    UCXTransportType = "cuda_ipc"
	UCXTransportTCP        UCXTransportType = "tcp"
)

// UCXTopologyEnvironment carries discovered container/host sysfs devices and operator overrides.
type UCXTopologyEnvironment struct {
	AvailableIBDevices []string          `json:"available_ib_devices"`
	ContainerSysfsOnly bool              `json:"container_sysfs_only"`
	OperatorOverrides  map[string]string `json:"operator_overrides"`
}

// UCXResolvedConfig reports the derived UCX parameters and decision provenance without mutating env.
type UCXResolvedConfig struct {
	Transport           UCXTransportType  `json:"transport"`
	NetDevices          string            `json:"net_devices"`
	TLS                 string            `json:"tls"`
	OperatorPrecedence  bool              `json:"operator_precedence"`
	ScrubbedSettings    map[string]string `json:"scrubbed_settings"`
	ResolutionRationale string            `json:"resolution_rationale"`
}

// ResolveUCXTopologyDefaults derives viable UCX transports and devices from hardware/container topology,
// strictly respecting explicit operator overrides with highest precedence without mutating environment variables.
func ResolveUCXTopologyDefaults(env UCXTopologyEnvironment) UCXResolvedConfig {
	scrubbed := make(map[string]string)
	for k, v := range env.OperatorOverrides {
		// Scrub any credentials if present
		if strings.Contains(strings.ToLower(k), "secret") || strings.Contains(strings.ToLower(k), "key") {
			scrubbed[k] = "[REDACTED]"
		} else {
			scrubbed[k] = v
		}
	}

	overrideTLS := env.OperatorOverrides["UCX_TLS"]
	overrideNetDevices := env.OperatorOverrides["UCX_NET_DEVICES"]

	// 1. Explicit operator override precedence
	if overrideTLS != "" || overrideNetDevices != "" {
		trans := UCXTransportTCP
		if strings.Contains(overrideTLS, "rc") || strings.Contains(overrideTLS, "ib") {
			trans = UCXTransportInfiniBand
		} else if strings.Contains(overrideTLS, "cuda_ipc") {
			trans = UCXTransportCUDAIPC
		}

		if overrideNetDevices == "" {
			overrideNetDevices = "all"
		}
		if overrideTLS == "" {
			overrideTLS = "tcp,cuda_copy"
		}

		return UCXResolvedConfig{
			Transport:           trans,
			NetDevices:          overrideNetDevices,
			TLS:                 overrideTLS,
			OperatorPrecedence:  true,
			ScrubbedSettings:    scrubbed,
			ResolutionRationale: "operator explicit override took precedence",
		}
	}

	// 2. Container sysfs-only environment check
	if env.ContainerSysfsOnly && len(env.AvailableIBDevices) > 0 {
		dev := env.AvailableIBDevices[0]
		tls := "rc_x,cuda_copy"
		scrubbed["UCX_TLS"] = tls
		scrubbed["UCX_NET_DEVICES"] = dev
		return UCXResolvedConfig{
			Transport:           UCXTransportInfiniBand,
			NetDevices:          dev,
			TLS:                 tls,
			OperatorPrecedence:  false,
			ScrubbedSettings:    scrubbed,
			ResolutionRationale: fmt.Sprintf("container sysfs-only detected with device %s", dev),
		}
	}

	// 3. Native host with InfiniBand/RoCE devices present
	if len(env.AvailableIBDevices) > 0 {
		sortedDevs := append([]string(nil), env.AvailableIBDevices...)
		sort.Strings(sortedDevs)
		devsStr := strings.Join(sortedDevs, ",")
		tls := "rc_x,cuda_copy,cuda_ipc"

		scrubbed["UCX_TLS"] = tls
		scrubbed["UCX_NET_DEVICES"] = devsStr

		return UCXResolvedConfig{
			Transport:           UCXTransportInfiniBand,
			NetDevices:          devsStr,
			TLS:                 tls,
			OperatorPrecedence:  false,
			ScrubbedSettings:    scrubbed,
			ResolutionRationale: fmt.Sprintf("native IB devices discovered: %s", devsStr),
		}
	}

	// 4. Default fallback to TCP
	tcpTLS := "tcp,cuda_copy"
	scrubbed["UCX_TLS"] = tcpTLS
	scrubbed["UCX_NET_DEVICES"] = "all"

	return UCXResolvedConfig{
		Transport:           UCXTransportTCP,
		NetDevices:          "all",
		TLS:                 tcpTLS,
		OperatorPrecedence:  false,
		ScrubbedSettings:    scrubbed,
		ResolutionRationale: "no IB devices present; safe fallback to TCP",
	}
}
