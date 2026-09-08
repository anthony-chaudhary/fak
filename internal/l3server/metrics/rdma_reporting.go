package metrics

import (
	"fmt"
	"strings"
)

func (c *Collector) writeRDMA(b *strings.Builder) {
	b.WriteString("\n\n# ---- RDMA --------------------------------------------------------\n\n")

	nicIPs := c.connReg.NICIPs()
	nicBytes := c.connReg.NICWireBytes()
	if len(nicIPs) > 0 {
		b.WriteString("# HELP l3_rdma_nic_wire_gb_total Per-NIC RDMA wire bytes in GiB.\n")
		b.WriteString("# TYPE l3_rdma_nic_wire_gb_total counter\n")
		for dev, ip := range nicIPs {
			total := nicBytes[dev]
			wireGB := float64(total[0]+total[1]) / (1 << 30)
			fmt.Fprintf(b, "l3_rdma_nic_wire_gb_total{device=\"%s\",ip=\"%s\"} %.6f\n", dev, ip, wireGB)
		}

		b.WriteString("\n# HELP l3_rdma_nic_wire_gb_recv Per-NIC RDMA bytes received in GiB.\n")
		b.WriteString("# TYPE l3_rdma_nic_wire_gb_recv counter\n")
		b.WriteString("# HELP l3_rdma_nic_wire_gb_sent Per-NIC RDMA bytes sent in GiB.\n")
		b.WriteString("# TYPE l3_rdma_nic_wire_gb_sent counter\n")
		for dev, ip := range nicIPs {
			total := nicBytes[dev]
			recvGB := float64(total[0]) / (1 << 30)
			sentGB := float64(total[1]) / (1 << 30)
			fmt.Fprintf(b, "l3_rdma_nic_wire_gb_recv{device=\"%s\",ip=\"%s\"} %.6f\n", dev, ip, recvGB)
			fmt.Fprintf(b, "l3_rdma_nic_wire_gb_sent{device=\"%s\",ip=\"%s\"} %.6f\n", dev, ip, sentGB)
		}

		// --- Aggregate RDMA wire totals across all NICs ---
		var aggRecvBytes, aggSentBytes int64
		for _, total := range nicBytes {
			aggRecvBytes += total[0]
			aggSentBytes += total[1]
		}
		b.WriteString("\n# HELP l3_rdma_wire_gb_recv_total Aggregate RDMA bytes received across all NICs in GiB.\n")
		b.WriteString("# TYPE l3_rdma_wire_gb_recv_total counter\n")
		fmt.Fprintf(b, "l3_rdma_wire_gb_recv_total %.6f\n", float64(aggRecvBytes)/(1<<30))

		b.WriteString("\n# HELP l3_rdma_wire_gb_sent_total Aggregate RDMA bytes sent across all NICs in GiB.\n")
		b.WriteString("# TYPE l3_rdma_wire_gb_sent_total counter\n")
		fmt.Fprintf(b, "l3_rdma_wire_gb_sent_total %.6f\n", float64(aggSentBytes)/(1<<30))

		b.WriteString("\n# HELP l3_rdma_wire_gb_total Aggregate RDMA wire bytes (recv+sent) across all NICs in GiB.\n")
		b.WriteString("# TYPE l3_rdma_wire_gb_total counter\n")
		fmt.Fprintf(b, "l3_rdma_wire_gb_total %.6f\n", float64(aggRecvBytes+aggSentBytes)/(1<<30))

		// --- Per-NIC throughput and saturation ---
		nicLinkRates := c.connReg.NICLinkRates()
		nicThroughput := c.connReg.NICThroughputGbps()

		b.WriteString("\n# HELP l3_rdma_nic_throughput_gbps Per-NIC bidirectional throughput in Gbps (averaged since epoch).\n")
		b.WriteString("# TYPE l3_rdma_nic_throughput_gbps gauge\n")
		for dev, ip := range nicIPs {
			fmt.Fprintf(b, "l3_rdma_nic_throughput_gbps{device=\"%s\",ip=\"%s\"} %.6f\n",
				dev, ip, nicThroughput[dev])
		}

		// Aggregate RDMA throughput across all NICs
		var totalTput float64
		for dev := range nicIPs {
			totalTput += nicThroughput[dev]
		}
		b.WriteString("\n# HELP l3_rdma_throughput_gbps Aggregate RDMA throughput across all NICs in Gbps.\n")
		b.WriteString("# TYPE l3_rdma_throughput_gbps gauge\n")
		fmt.Fprintf(b, "l3_rdma_throughput_gbps %.6f\n", totalTput)

		if len(nicLinkRates) > 0 {
			b.WriteString("\n# HELP l3_rdma_nic_link_rate_gbps Detected RDMA link rate per NIC in Gbps.\n")
			b.WriteString("# TYPE l3_rdma_nic_link_rate_gbps gauge\n")
			for dev, ip := range nicIPs {
				fmt.Fprintf(b, "l3_rdma_nic_link_rate_gbps{device=\"%s\",ip=\"%s\"} %.0f\n",
					dev, ip, nicLinkRates[dev])
			}

			// Aggregate link capacity
			var totalCap float64
			for dev := range nicIPs {
				totalCap += nicLinkRates[dev]
			}
			b.WriteString("\n# HELP l3_rdma_link_rate_gbps_total Aggregate RDMA link capacity across all NICs in Gbps.\n")
			b.WriteString("# TYPE l3_rdma_link_rate_gbps_total gauge\n")
			fmt.Fprintf(b, "l3_rdma_link_rate_gbps_total %.0f\n", totalCap)

			b.WriteString("\n# HELP l3_rdma_nic_saturation_pct Per-NIC wire saturation as a percentage of link rate (0-100).\n")
			b.WriteString("# TYPE l3_rdma_nic_saturation_pct gauge\n")
			var minSat, maxSat float64
			first := true
			for dev, ip := range nicIPs {
				rate := nicLinkRates[dev]
				tput := nicThroughput[dev]
				pct := 0.0
				if rate > 0 {
					pct = tput / rate * 100
					if pct > 100 {
						pct = 100
					}
				}
				fmt.Fprintf(b, "l3_rdma_nic_saturation_pct{device=\"%s\",ip=\"%s\"} %.2f\n",
					dev, ip, pct)
				if first || pct < minSat {
					minSat = pct
				}
				if first || pct > maxSat {
					maxSat = pct
				}
				first = false
			}

			balance := 100.0
			if maxSat > 0 {
				balance = minSat / maxSat * 100
			}
			b.WriteString("\n# HELP l3_rdma_nic_balance_pct Cross-NIC traffic balance (100=perfectly equal, low=skewed).\n")
			b.WriteString("# TYPE l3_rdma_nic_balance_pct gauge\n")
			fmt.Fprintf(b, "l3_rdma_nic_balance_pct %.2f\n", balance)

			wireSat := 0.0
			if totalCap > 0 {
				wireSat = totalTput / totalCap * 100
			}
			b.WriteString("\n# HELP l3_wire_saturation_pct Aggregate wire saturation across all NICs as a percentage of total link capacity (0-100).\n")
			b.WriteString("# TYPE l3_wire_saturation_pct gauge\n")
			fmt.Fprintf(b, "l3_wire_saturation_pct %.2f\n", wireSat)
		}
	}

	if c.PollerMetrics != nil {
		psnaps := c.PollerMetrics()
		if len(psnaps) > 0 {
			b.WriteString("\n# HELP l3_rdma_poller_active_conns Active connections on this CQ poller.\n")
			b.WriteString("# TYPE l3_rdma_poller_active_conns gauge\n")
			b.WriteString("# HELP l3_rdma_poller_completions_total Total CQ completions polled.\n")
			b.WriteString("# TYPE l3_rdma_poller_completions_total counter\n")
			b.WriteString("# HELP l3_rdma_poller_dispatch_enqueued_total Items enqueued to dispatch workers.\n")
			b.WriteString("# TYPE l3_rdma_poller_dispatch_enqueued_total counter\n")
			b.WriteString("# HELP l3_rdma_poller_dispatch_dropped_total Items dropped (dispatch queue full).\n")
			b.WriteString("# TYPE l3_rdma_poller_dispatch_dropped_total counter\n")
			b.WriteString("# HELP l3_rdma_poller_send_ch_dropped_total Responses dropped (send channel full).\n")
			b.WriteString("# TYPE l3_rdma_poller_send_ch_dropped_total counter\n")
			b.WriteString("# HELP l3_rdma_poller_dispatch_workers Number of dispatch workers.\n")
			b.WriteString("# TYPE l3_rdma_poller_dispatch_workers gauge\n")
			b.WriteString("# HELP l3_rdma_poller_dispatch_queue_depth Pending items in RDMA dispatch queue.\n")
			b.WriteString("# TYPE l3_rdma_poller_dispatch_queue_depth gauge\n")
			b.WriteString("# HELP l3_rdma_poller_dispatch_queue_capacity RDMA dispatch queue capacity.\n")
			b.WriteString("# TYPE l3_rdma_poller_dispatch_queue_capacity gauge\n")
			b.WriteString("# HELP l3_rdma_cleanup_queue_depth Pending items in RDMA cleanup queue.\n")
			b.WriteString("# TYPE l3_rdma_cleanup_queue_depth gauge\n")
			b.WriteString("# HELP l3_rdma_cleanup_queue_capacity RDMA cleanup queue capacity.\n")
			b.WriteString("# TYPE l3_rdma_cleanup_queue_capacity gauge\n")
			b.WriteString("# HELP l3_rdma_poller_dispatch_saturation_pct Dispatch queue saturation (0-100).\n")
			b.WriteString("# TYPE l3_rdma_poller_dispatch_saturation_pct gauge\n")
			for _, ps := range psnaps {
				fmt.Fprintf(b, "l3_rdma_poller_active_conns{device=\"%s\"} %d\n", ps.Device, ps.ActiveConns)
				fmt.Fprintf(b, "l3_rdma_poller_completions_total{device=\"%s\"} %d\n", ps.Device, ps.Completions)
				fmt.Fprintf(b, "l3_rdma_poller_dispatch_enqueued_total{device=\"%s\"} %d\n", ps.Device, ps.DispatchEnqueued)
				fmt.Fprintf(b, "l3_rdma_poller_dispatch_dropped_total{device=\"%s\"} %d\n", ps.Device, ps.DispatchDropped)
				fmt.Fprintf(b, "l3_rdma_poller_send_ch_dropped_total{device=\"%s\"} %d\n", ps.Device, ps.SendChDropped)
				fmt.Fprintf(b, "l3_rdma_poller_dispatch_workers{device=\"%s\"} %d\n", ps.Device, ps.DispatchWorkers)
				fmt.Fprintf(b, "l3_rdma_poller_dispatch_queue_depth{device=\"%s\"} %d\n", ps.Device, ps.DispatchQueueDepth)
				fmt.Fprintf(b, "l3_rdma_poller_dispatch_queue_capacity{device=\"%s\"} %d\n", ps.Device, ps.DispatchQueueCap)
				fmt.Fprintf(b, "l3_rdma_cleanup_queue_depth{device=\"%s\"} %d\n", ps.Device, ps.CleanupQueueDepth)
				fmt.Fprintf(b, "l3_rdma_cleanup_queue_capacity{device=\"%s\"} %d\n", ps.Device, ps.CleanupQueueCap)
				fmt.Fprintf(b, "l3_rdma_poller_dispatch_saturation_pct{device=\"%s\"} %.2f\n", ps.Device, ps.DispatchSaturationPct)
			}
		}
	}

	if c.ReplicationQueue != nil {
		rqDepth, rqCap := c.ReplicationQueue()
		b.WriteString("\n# HELP l3_cluster_replication_queue_depth Pending items in cluster replication queue.\n")
		b.WriteString("# TYPE l3_cluster_replication_queue_depth gauge\n")
		fmt.Fprintf(b, "l3_cluster_replication_queue_depth %d\n", rqDepth)
		b.WriteString("# HELP l3_cluster_replication_queue_capacity Cluster replication queue capacity.\n")
		b.WriteString("# TYPE l3_cluster_replication_queue_capacity gauge\n")
		fmt.Fprintf(b, "l3_cluster_replication_queue_capacity %d\n", rqCap)
	}

	if c.RDMAReadMetrics != nil {
		rs := c.RDMAReadMetrics()
		b.WriteString("\n# HELP l3_rdma_reads_issued_total RDMA Read operations issued.\n")
		b.WriteString("# TYPE l3_rdma_reads_issued_total counter\n")
		fmt.Fprintf(b, "l3_rdma_reads_issued_total %d\n", rs.Issued)

		b.WriteString("\n# HELP l3_rdma_reads_confirmed_total RDMA Read operations confirmed by client.\n")
		b.WriteString("# TYPE l3_rdma_reads_confirmed_total counter\n")
		fmt.Fprintf(b, "l3_rdma_reads_confirmed_total %d\n", rs.Confirmed)

		b.WriteString("\n# HELP l3_rdma_reads_failed_total RDMA Read operations that failed.\n")
		b.WriteString("# TYPE l3_rdma_reads_failed_total counter\n")
		fmt.Fprintf(b, "l3_rdma_reads_failed_total %d\n", rs.Failed)

		b.WriteString("\n# HELP l3_rdma_reads_forced_inline_total GETs forced inline during migration.\n")
		b.WriteString("# TYPE l3_rdma_reads_forced_inline_total counter\n")
		fmt.Fprintf(b, "l3_rdma_reads_forced_inline_total %d\n", rs.ForcedInline)

		b.WriteString("\n# HELP l3_rdma_lease_drops_total RDMA leases dropped (shard channel full), forced inline fallback.\n")
		b.WriteString("# TYPE l3_rdma_lease_drops_total counter\n")
		fmt.Fprintf(b, "l3_rdma_lease_drops_total %d\n", rs.LeaseDrops)

		b.WriteString("\n# HELP l3_rdma_mget_migration_skips_total MGET RDMA keys skipped because their shard was migrating.\n")
		b.WriteString("# TYPE l3_rdma_mget_migration_skips_total counter\n")
		fmt.Fprintf(b, "l3_rdma_mget_migration_skips_total %d\n", rs.MgetMigrationSkips)
	}
}

func (c *Collector) writePerConnection(b *strings.Builder) {
	conns := c.connReg.Snapshot()
	if len(conns) == 0 {
		return
	}

	b.WriteString("\n\n# ---- per-connection ----------------------------------------------\n\n")

	b.WriteString("# HELP l3_conn_bytes_recv Bytes received per connection (label 'remote' contains client IP:port).\n")
	b.WriteString("# TYPE l3_conn_bytes_recv counter\n")
	b.WriteString("# HELP l3_conn_bytes_sent Bytes sent per connection.\n")
	b.WriteString("# TYPE l3_conn_bytes_sent counter\n")
	b.WriteString("# HELP l3_conn_requests Requests handled per connection.\n")
	b.WriteString("# TYPE l3_conn_requests counter\n")
	for _, cm := range conns {
		fmt.Fprintf(b, "l3_conn_bytes_recv{transport=\"%s\",remote=\"%s\",device=\"%s\"} %d\n",
			cm.Transport, cm.RemoteAddr, cm.Device, cm.BytesRecv())
		fmt.Fprintf(b, "l3_conn_bytes_sent{transport=\"%s\",remote=\"%s\",device=\"%s\"} %d\n",
			cm.Transport, cm.RemoteAddr, cm.Device, cm.BytesSent())
		fmt.Fprintf(b, "l3_conn_requests{transport=\"%s\",remote=\"%s\",device=\"%s\"} %d\n",
			cm.Transport, cm.RemoteAddr, cm.Device, cm.Requests())
	}
}
