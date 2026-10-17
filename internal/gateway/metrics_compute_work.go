package gateway

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// writeComputeWorkMetrics renders where native work ran: device dispatches per
// initialized backend, host fallbacks per instrumented stage, and device-to-host
// readback bytes. A backend or stage with no instrumentation has no series, so
// absent means unmeasured, never a measured zero.
func writeComputeWorkMetrics(b *strings.Builder) {
	devices := compute.DeviceWorkSnapshot()
	writeHelpType(b, "fak_compute_device_dispatches_total", "Compute dispatches recorded on a device backend since process start.", "counter")
	for _, d := range devices {
		fmt.Fprintf(b, "fak_compute_device_dispatches_total{backend=\"%s\"} %d\n", promQuote(d.Backend), d.Dispatches)
	}
	writeHelpType(b, "fak_compute_host_fallback_total", "Native decode-path stages that ran on the host where a device path was expected.", "counter")
	for _, c := range compute.HostFallbackCounts() {
		fmt.Fprintf(b, "fak_compute_host_fallback_total{stage=\"%s\"} %d\n", promQuote(c.Stage), c.Count)
	}
	writeHelpType(b, "fak_compute_readback_bytes_total", "Device-to-host readback payload bytes across device backends since process start.", "counter")
	if len(devices) > 0 {
		var total uint64
		for _, d := range devices {
			total += d.ReadbackBytes
		}
		fmt.Fprintf(b, "fak_compute_readback_bytes_total %d\n", total)
	}
}
