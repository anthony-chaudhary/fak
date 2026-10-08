package metrics

import (
	"fmt"
	"strings"
	"time"
)

func (c *Collector) writeSystemHealth(b *strings.Builder) {
	if c.SystemHealth == nil {
		return
	}
	h := c.SystemHealth()

	b.WriteString("\n\n# ---- system health -----------------------------------------------\n\n")

	b.WriteString("# HELP l3_system_memory_available_bytes Available system memory in bytes.\n")
	b.WriteString("# TYPE l3_system_memory_available_bytes gauge\n")
	fmt.Fprintf(b, "l3_system_memory_available_bytes %d\n", h.MemAvailableBytes)

	memPressureVal := 0
	if h.MemPressureActive {
		memPressureVal = 1
	}
	b.WriteString("\n# HELP l3_system_memory_pressure_active Whether the memory pressure circuit breaker is engaged (1=active, 0=normal).\n")
	b.WriteString("# TYPE l3_system_memory_pressure_active gauge\n")
	fmt.Fprintf(b, "l3_system_memory_pressure_active %d\n", memPressureVal)

	b.WriteString("\n# HELP l3_system_memory_pressure_level Tiered memory pressure level (0=normal, 1=elevated, 2=high, 3=critical, 4=emergency).\n")
	b.WriteString("# TYPE l3_system_memory_pressure_level gauge\n")
	fmt.Fprintf(b, "l3_system_memory_pressure_level %d\n", h.MemPressureLevel)

	b.WriteString("\n# HELP l3_system_memory_psi_some_bp Memory PSI 'some' pressure in basis points.\n")
	b.WriteString("# TYPE l3_system_memory_psi_some_bp gauge\n")
	fmt.Fprintf(b, "l3_system_memory_psi_some_bp %d\n", h.MemPSISomeBP)

	b.WriteString("\n# HELP l3_system_memory_psi_full_bp Memory PSI 'full' pressure in basis points.\n")
	b.WriteString("# TYPE l3_system_memory_psi_full_bp gauge\n")
	fmt.Fprintf(b, "l3_system_memory_psi_full_bp %d\n", h.MemPSIFullBP)

	// Memory accounting: Linux splits process memory into two pools.
	// VmRSS (/proc/self/status) counts regular 4KB pages ONLY â€” it excludes
	// hugepage-backed memory. Slab memory on hugepages shows up exclusively in
	// smaps_rollup (Private_Hugetlb + Shared_Hugetlb). htop's RES column reads
	// /proc/[pid]/statm which includes BOTH, so htop shows a much larger number
	// than VmRSS alone. Use l3_system_process_total_rss_bytes to match htop.
	b.WriteString("\n# HELP l3_system_process_total_rss_bytes Total process resident memory in bytes (VmRSS + HugeTLB). Matches htop RES column. This is the metric you want for dashboards and alerts.\n")
	b.WriteString("# TYPE l3_system_process_total_rss_bytes gauge\n")
	fmt.Fprintf(b, "l3_system_process_total_rss_bytes %d\n", h.ProcessTotalRSSBytes)

	b.WriteString("\n# HELP l3_system_process_rss_bytes VmRSS in bytes (regular 4KB pages ONLY). Does NOT include hugeTLB-backed slab memory â€” on a healthy system with hugepages this will be much smaller than htop shows. See l3_system_process_total_rss_bytes for the full picture.\n")
	b.WriteString("# TYPE l3_system_process_rss_bytes gauge\n")
	fmt.Fprintf(b, "l3_system_process_rss_bytes %d\n", h.ProcessRSSBytes)

	b.WriteString("\n# HELP l3_system_process_hugetlb_bytes Hugepage-backed memory in bytes (Private_Hugetlb + Shared_Hugetlb from smaps_rollup). This is where slab memory lives when hugepages are working. 0 means hugepages are not in use or kernel < 4.14.\n")
	b.WriteString("# TYPE l3_system_process_hugetlb_bytes gauge\n")
	fmt.Fprintf(b, "l3_system_process_hugetlb_bytes %d\n", h.ProcessHugetlbBytes)

	if h.FDLimit > 0 {
		fdRatio := float64(h.FDCount) / float64(h.FDLimit)
		b.WriteString("\n# HELP l3_system_fd_ratio File descriptor usage ratio (count/limit).\n")
		b.WriteString("# TYPE l3_system_fd_ratio gauge\n")
		fmt.Fprintf(b, "l3_system_fd_ratio %.4f\n", fdRatio)
	}

	if len(h.NICPortActive) > 0 {
		b.WriteString("\n# HELP l3_rdma_nic_port_active Whether the RDMA NIC port is active (1=active, 0=down).\n")
		b.WriteString("# TYPE l3_rdma_nic_port_active gauge\n")
		for dev, active := range h.NICPortActive {
			val := 0
			if active {
				val = 1
			}
			fmt.Fprintf(b, "l3_rdma_nic_port_active{device=\"%s\"} %d\n", dev, val)
		}
	}

	if len(h.NICHWErrors) > 0 {
		b.WriteString("# HELP l3_rdma_nic_hw_errors_total Accumulated RDMA NIC hardware errors.\n")
		b.WriteString("# TYPE l3_rdma_nic_hw_errors_total counter\n")
		for dev, errs := range h.NICHWErrors {
			for errType, count := range errs {
				if count > 0 {
					fmt.Fprintf(b, "l3_rdma_nic_hw_errors_total{device=\"%s\",type=\"%s\"} %d\n", dev, errType, count)
				}
			}
		}
	}

	// CPU metrics
	b.WriteString("\n# HELP l3_system_cpu_seconds_total Cumulative process CPU time in seconds.\n")
	b.WriteString("# TYPE l3_system_cpu_seconds_total counter\n")
	fmt.Fprintf(b, "l3_system_cpu_seconds_total{mode=\"user\"} %.2f\n", h.CPUUserSeconds)
	fmt.Fprintf(b, "l3_system_cpu_seconds_total{mode=\"system\"} %.2f\n", h.CPUSystemSeconds)

	b.WriteString("\n# HELP l3_system_goroutines Current number of goroutines.\n")
	b.WriteString("# TYPE l3_system_goroutines gauge\n")
	fmt.Fprintf(b, "l3_system_goroutines %d\n", h.Goroutines)

	b.WriteString("\n# HELP l3_system_threads Current number of OS threads.\n")
	b.WriteString("# TYPE l3_system_threads gauge\n")
	fmt.Fprintf(b, "l3_system_threads %d\n", h.Threads)

	b.WriteString("\n# HELP l3_system_context_switches_total Cumulative context switches.\n")
	b.WriteString("# TYPE l3_system_context_switches_total counter\n")
	fmt.Fprintf(b, "l3_system_context_switches_total{type=\"voluntary\"} %d\n", h.VoluntaryCtxSwitches)
	fmt.Fprintf(b, "l3_system_context_switches_total{type=\"involuntary\"} %d\n", h.InvoluntaryCtxSwitches)
}

func (c *Collector) writeVacuum(b *strings.Builder) {
	if c.VacuumMetrics == nil {
		return
	}
	vs := c.VacuumMetrics()

	b.WriteString("\n\n# ---- vacuum coordinator ------------------------------------------\n\n")

	b.WriteString("# HELP l3_vacuum_rebalances_total Total vacuum-triggered rebalances.\n")
	b.WriteString("# TYPE l3_vacuum_rebalances_total counter\n")
	fmt.Fprintf(b, "l3_vacuum_rebalances_total %d\n", vs.RebalancesTotal)

	if vs.LastRebalanceEpoch > 0 {
		b.WriteString("\n# HELP l3_vacuum_last_rebalance_seconds Seconds since the last rebalance.\n")
		b.WriteString("# TYPE l3_vacuum_last_rebalance_seconds gauge\n")
		fmt.Fprintf(b, "l3_vacuum_last_rebalance_seconds %.0f\n",
			time.Since(time.Unix(vs.LastRebalanceEpoch, 0)).Seconds())
	}

	b.WriteString("\n# HELP l3_vacuum_pending_shards Shards waiting for rebalance.\n")
	b.WriteString("# TYPE l3_vacuum_pending_shards gauge\n")
	fmt.Fprintf(b, "l3_vacuum_pending_shards %d\n", vs.PendingShards)

	b.WriteString("\n# HELP l3_vacuum_pressure_evals_total Total pressure evaluation cycles.\n")
	b.WriteString("# TYPE l3_vacuum_pressure_evals_total counter\n")
	fmt.Fprintf(b, "l3_vacuum_pressure_evals_total %d\n", vs.PressureEvals)

	b.WriteString("\n# HELP l3_vacuum_pressure_rebuilds_total Rebuilds triggered by pressure.\n")
	b.WriteString("# TYPE l3_vacuum_pressure_rebuilds_total counter\n")
	fmt.Fprintf(b, "l3_vacuum_pressure_rebuilds_total %d\n", vs.PressureRebuilds)

	b.WriteString("\n# HELP l3_vacuum_max_drift Maximum weight drift across slab classes.\n")
	b.WriteString("# TYPE l3_vacuum_max_drift gauge\n")
	fmt.Fprintf(b, "l3_vacuum_max_drift %.4f\n", vs.MaxDrift)

	b.WriteString("\n# HELP l3_vacuum_rebalance_failures_total Failed rebalance attempts.\n")
	b.WriteString("# TYPE l3_vacuum_rebalance_failures_total counter\n")
	fmt.Fprintf(b, "l3_vacuum_rebalance_failures_total %d\n", vs.RebalanceFailures)
}
