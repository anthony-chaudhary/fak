package perfledger

import "testing"

// TestMemModel verifies the memory-estimation model returns the exact
// footprint the aiperf source (ai-dynamo/aiperf@ab610add,
// src/aiperf/kubernetes/_memory_estimator) produces for known workload params.
// Every expected value below is a hand-computed source output: the constants
// and formulas are mirrored verbatim, so the Go results must match the source
// bit-for-bit (IEEE 754 double, same operation order).
func TestMemModel(t *testing.T) {
	// --- helpers / pure math -------------------------------------------------
	if got := ceilPow2(1000); got != 1024 {
		t.Errorf("ceilPow2(1000) = %d, want 1024", got)
	}
	if got := ceilPow2(127000); got != 131072 {
		t.Errorf("ceilPow2(127000) = %d, want 131072", got)
	}
	if got := ceilDiv(100, 10); got != 10 {
		t.Errorf("ceilDiv(100,10) = %d, want 10", got)
	}
	if got := ceilDiv(100, 2); got != 50 {
		t.Errorf("ceilDiv(100,2) = %d, want 50", got)
	}
	if got := ceilDiv(50, 2); got != 25 {
		t.Errorf("ceilDiv(50,2) = %d, want 25", got)
	}
	if got := rpQueueDepth(25, 512, 128); got != 25 {
		t.Errorf("rpQueueDepth(25,512,128) = %d, want 25", got)
	}

	// --- per-request bytes ---------------------------------------------------
	if got := perRequestBytes(512, 128, true); got != 63320 {
		t.Errorf("perRequestBytes(512,128,streaming) = %d, want 63320", got)
	}
	if got := perRequestBytes(512, 128, false); got != 8552 {
		t.Errorf("perRequestBytes(512,128,text) = %d, want 8552", got)
	}

	// --- fixed services ------------------------------------------------------
	checkFixed := func(name string, wantBase float64) {
		t.Helper()
		got := estimateFixedService(name)
		if got.BaseMiB != wantBase || got.VariableMiB != 0 || got.PeakMiB != wantBase {
			t.Errorf("estimateFixedService(%q) = (%v,%v,%v), want (%v,0,%v)",
				name, got.BaseMiB, got.VariableMiB, got.PeakMiB, wantBase, wantBase)
		}
	}
	checkFixed("system_controller", 175)
	checkFixed("timing_manager", 165)
	checkFixed("api_service", 170)
	checkFixed("results_sidecar", 160)
	checkFixed("worker_group_manager", 160)

	// --- dataset manager -----------------------------------------------------
	dm := estimateDatasetManager(100, 512, 128, 1)
	assertMiB(t, "dataset_manager.base", dm.BaseMiB, 180)
	assertMiB(t, "dataset_manager.variable", dm.VariableMiB, 0.00152587890625)
	assertMiB(t, "dataset_manager.peak", dm.PeakMiB, 181.11961364746094)

	// --- records manager -----------------------------------------------------
	rm := estimateRecordsManager(1000, 25, 128, true, "ragged", 100)
	assertMiB(t, "records_manager.base", rm.BaseMiB, 190)
	assertMiB(t, "records_manager.variable", rm.VariableMiB, 3.006439208984375)
	assertMiB(t, "records_manager.peak", rm.PeakMiB, 193.3070831298828)

	// --- worker --------------------------------------------------------------
	w := estimateWorker(10, 128, true, 1, 512, 100)
	assertMiB(t, "worker.base", w.BaseMiB, 162)
	assertMiB(t, "worker.variable", w.VariableMiB, 0.7015228271484375)
	assertMiB(t, "worker.peak", w.PeakMiB, 162.77167510986328)

	// --- record processor ----------------------------------------------------
	rp := estimateRecordProcessor(1, 512, 128, true, 25)
	assertMiB(t, "record_processor.base", rp.BaseMiB, 160)
	assertMiB(t, "record_processor.variable", rp.VariableMiB, 152.15621948242188)
	assertMiB(t, "record_processor.peak", rp.PeakMiB, 327.37184143066406)

	raw, exp := rpBufferMiB(128, true)
	assertMiB(t, "rp_buffer.raw", raw, 0.5416488647460938)
	assertMiB(t, "rp_buffer.export", exp, 0.1049041748046875)

	// --- full cluster estimate ----------------------------------------------
	p := Params{
		TotalWorkers:                 10,
		WorkersPerPod:                5,
		NumWorkerPods:                2,
		RecordProcessorsPerPod:       2,
		MaxConcurrency:               100,
		TotalRequests:                1000,
		TotalBenchmarkDurationS:      3600,
		DatasetCount:                 100,
		AvgISL:                       512,
		AvgOSL:                       128,
		MaxTurns:                     1,
		Streaming:                    true,
		ListMetricBackend:            "ragged",
		ConnectionsPerWorker:         100,
		GPUTelemetryEnabled:          false,
		ServerMetricsEnabled:         false,
		NumGPUs:                      0,
		GPUSampleIntervalS:           1.0,
		NumGPUMetrics:                12,
		NumServerMetricsEndpoints:    0,
		ServerMetricsScrapeIntervalS: 5.0,
		EstUniqueMetricSeries:        200,
		EstHistogramMetrics:          20,
		EstHistogramBuckets:          10,
		NumModels:                    1,
		NumStandardMetrics:           25,
	}
	est := Estimate(p)

	assertMiB(t, "controller.steady", est.Controller.TotalSteadyStateMiB(), 1058.0079650878906)
	assertMiB(t, "controller.peak", est.Controller.TotalPeakMiB(), 1059.4266967773438)
	assertMiB(t, "worker_pod.steady", est.WorkerPod.TotalSteadyStateMiB(), 1597.820053100586)
	assertMiB(t, "worker_pod.peak", est.WorkerPod.TotalPeakMiB(), 1628.6020584106445)
	assertMiB(t, "operator.steady", est.Operator.TotalSteadyStateMiB(), 256.0)
	assertMiB(t, "operator.peak", est.Operator.TotalPeakMiB(), 256.0)

	// The headline footprint: cluster-wide steady-state and peak.
	assertMiB(t, "total_cluster.steady", est.TotalClusterMiB(), 4509.6480712890625)
	assertMiB(t, "total_cluster.peak", est.TotalClusterPeakMiB(), 4572.630813598633)

	// Recommended sizing (margins applied to the controller pod).
	if got := est.Controller.RecommendedRequestMiB(); got != 1270 {
		t.Errorf("controller.RecommendedRequestMiB() = %d, want 1270", got)
	}
	if got := est.Controller.RecommendedLimitMiB(); got != 1378 {
		t.Errorf("controller.RecommendedLimitMiB() = %d, want 1378", got)
	}
}

// assertMiB checks an exact float64 MiB value, reporting both sides on a miss.
func assertMiB(t *testing.T, label string, got, want float64) {
	t.Helper()
	if got != want {
		t.Errorf("%s: got %v MiB, want %v MiB (diff %v)", label, got, want, got-want)
	}
}
