// NOTICE / attribution:
// Adapted from ai-dynamo/aiperf, src/aiperf/kubernetes/_memory_estimator/
// (constants.py, utils.py, components.py, estimates.py, estimator.py) at
// ab610add4e8876fca223197596c728e17365f381. License: Apache-2.0
// (SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.).
//
// Re-expressed in idiomatic Go. The constant values and estimation math are
// mirrored verbatim from the source. The source estimates the host-side (CPU)
// memory footprint of an AIPerf benchmark deployment — Python subprocess
// baselines, per-service overhead, and per-in-flight-request bytes — not
// GPU model-weight memory. Params therefore mirror the source's
// MemoryEstimationParams (workload shape: requests, ISL/OSL, concurrency,
// topology), not model size/dtype/batch/ctx.
package perfledger

import (
	"fmt"
	"math"
	"math/bits"
)

// ---------------------------------------------------------------------------
// Calibration constants (mirrored from aiperf constants.py).
//
// Two classes, as in the source:
//   - Aggregate MiB baselines (subprocess/service/tokenizer baselines, margin
//     multipliers) calibrated against real-cluster working-set sweeps.
//   - Per-object BYTE constants measured in-process against the real model
//     classes.
//
// The full source constant set is carried for fidelity; a few entries are
// defaults or warning thresholds the ported formulas do not consume.
// ---------------------------------------------------------------------------

const (
	pythonSubprocessBaseMiB      = 150
	pythonChildSubprocessBaseMiB = 150

	zmqProxyMiB   = 5
	numZMQProxies = 3

	growableArrayOverhead = 1.05

	float64Bytes = 8
	int32Bytes   = 4
	int64Bytes   = 8

	tokenizerCacheMiB  = 150
	bytesPerConnection = 1024

	requestRecordBaseBytes    = 3600
	sseMessageBaseBytes       = 136
	sseBytesPerChunk          = 432
	textResponseBaseBytes     = 152
	textResponseBytesPerToken = 4
	turnBaseBytes             = 2240
	turnBytesPerToken         = 4
	mmapIndexEntryBytes       = 16

	defaultGPUMetrics         = 12
	defaultScrapeIntervalS    = 5.0
	defaultUniqueMetricSeries = 200
	defaultHistogramMetrics   = 20
	defaultHistogramBuckets   = 10

	steadyStateMargin     = 1.2
	peakMargin            = 1.3
	headroomWarningPct    = 15.0
	recordsManagerWarnPct = 50.0
	adequateHeadroomPct   = 30.0

	defaultNumStandardMetrics = 25

	columnStoreListMetricColumns          = 1
	columnStoreTimestampColumns           = 3
	columnStoreMetadataNumericColumns     = 4
	columnStoreMetadataCategoricalColumns = 6
	columnStoreMetadataBoolColumns        = 2
	columnStoreInitialCapacity            = 1024
	raggedOffsetsInitialCapacity          = 256
	categoricalInternBytesPerRequest      = 136
	tdigestListBackendBytes               = 4 * 1024
	defaultPhaseRequestCount              = 1000
	phaseAvgSecPerRequest                 = 2.0

	// components.py module-local calibration constants.
	rawBatchSize           = 10
	exportBatchSize        = 100
	exportBytesPerRecord   = 1100
	bytesPerTokenInDataset = 16
	turnOverheadBytes      = 1500
)

// serviceBaseMiB mirrors _SERVICE_BASE_MIB: per-service base overhead beyond
// the subprocess baseline.
var serviceBaseMiB = map[string]int{
	"system_controller":      25,
	"worker_manager":         15,
	"timing_manager":         15,
	"dataset_manager":        30,
	"records_manager":        40,
	"api_service":            20,
	"gpu_telemetry_manager":  15,
	"server_metrics_manager": 15,
	"results_sidecar":        10,
	"worker":                 12,
	"record_processor":       10,
	"worker_group_manager":   10,
}

// ceilPow2 returns the next power of 2 >= n (matches GrowableArray doubling
// behavior). n <= 0 yields 1.
func ceilPow2(n int) int {
	if n <= 0 {
		return 1
	}
	return 1 << bits.Len(uint(n-1))
}

// mib converts a byte count to MiB.
func mib(byteCount float64) float64 {
	return byteCount / (1024 * 1024)
}

// ceilDiv returns the maximum whole-number share across denominator members.
func ceilDiv(numerator, denominator int) int {
	d := denominator
	if d < 1 {
		d = 1
	}
	q := (numerator + d - 1) / d
	if q < 1 {
		q = 1
	}
	return q
}

// rpQueueDepth models the record-processor pull-queue depth under token
// pressure: above 10K tokens/request, tokenization becomes the bottleneck and
// records pile up in the (effectively unbounded) ZMQ pull queue.
func rpQueueDepth(concPerRP, avgISL, avgOSL int) int {
	avgTokens := avgISL + avgOSL
	if avgTokens > 10000 {
		pressure := float64(avgTokens) / 10000
		if pressure > 10.0 {
			pressure = 10.0
		}
		return int(float64(concPerRP) * pressure)
	}
	return concPerRP
}

// perRequestBytes is the bytes held in memory for a single in-flight
// RequestRecord: the record shell, its turn (prompt), and its response (one
// SSEMessage per streamed chunk, or a single TextResponse holding the whole
// buffered body).
func perRequestBytes(avgISL, avgOSL int, streaming bool) int {
	turnBytes := turnBaseBytes + avgISL*turnBytesPerToken
	var responseBytes int
	if streaming {
		responseBytes = sseMessageBaseBytes + avgOSL*sseBytesPerChunk
	} else {
		responseBytes = textResponseBaseBytes + avgOSL*textResponseBytesPerToken
	}
	return requestRecordBaseBytes + turnBytes + responseBytes
}

// ComponentEstimate is the memory estimate for one logical component.
type ComponentEstimate struct {
	Name        string
	BaseMiB     float64
	VariableMiB float64
	PeakMiB     float64
}

// SteadyStateMiB is base + variable.
func (c ComponentEstimate) SteadyStateMiB() float64 {
	return c.BaseMiB + c.VariableMiB
}

// estimateRecordsManager estimates the RecordsManager: it accumulates metric
// and metadata columns for the whole run.
func estimateRecordsManager(totalRequests, numMetrics, avgOSL int, streaming bool, listMetricBackend string, datasetCount int) ComponentEstimate {
	base := float64(serviceBaseMiB["records_manager"] + pythonSubprocessBaseMiB)

	capacity := columnStoreInitialCapacity
	if cp := ceilPow2(totalRequests); cp > capacity {
		capacity = cp
	}
	scalarMetrics := numMetrics - columnStoreListMetricColumns
	if scalarMetrics < 0 {
		scalarMetrics = 0
	}
	float64Columns := scalarMetrics + columnStoreTimestampColumns + columnStoreMetadataNumericColumns
	fixedColumnBytesPerRow := float64Columns*float64Bytes +
		columnStoreMetadataCategoricalColumns*int32Bytes +
		columnStoreMetadataBoolColumns
	columnBytes := float64(capacity*fixedColumnBytesPerRow) * growableArrayOverhead

	internEntries := totalRequests + min(totalRequests, max(datasetCount, 0))
	internBytes := internEntries * categoricalInternBytesPerRequest

	var listBytes float64
	if streaming && listMetricBackend == "ragged" {
		iclValues := totalRequests * max(avgOSL-1, 0)
		if iclValues > 0 {
			valuesCapacity := columnStoreInitialCapacity
			if cp := ceilPow2(iclValues); cp > valuesCapacity {
				valuesCapacity = cp
			}
			offsetsCapacity := raggedOffsetsInitialCapacity
			if cp := ceilPow2(totalRequests); cp > offsetsCapacity {
				offsetsCapacity = cp
			}
			listBytes = float64(valuesCapacity*(float64Bytes+int32Bytes))*growableArrayOverhead +
				float64(offsetsCapacity*int64Bytes)
		}
	} else if streaming && avgOSL > 1 {
		listBytes = float64(tdigestListBackendBytes)
	}

	// Per-worker tracking overhead (small, scales with workers not requests).
	trackerMiB := 1.0
	columnMiB := mib(columnBytes)
	internMiB := mib(float64(internBytes))
	listMiB := mib(listBytes)
	variable := columnMiB + internMiB + listMiB + trackerMiB
	peak := base + variable*1.1 // 10% finalization overhead

	return ComponentEstimate{Name: "RecordsManager", BaseMiB: base, VariableMiB: variable, PeakMiB: peak}
}

// estimateDatasetManager estimates the DatasetManager: generation peak holds
// the full dataset as conversation objects; steady state is mmap metadata only.
func estimateDatasetManager(datasetCount, avgISL, avgOSL, maxTurns int) ComponentEstimate {
	base := float64(serviceBaseMiB["dataset_manager"] + pythonSubprocessBaseMiB)

	bytesPerTurn := turnOverheadBytes + (avgISL+avgOSL)*bytesPerTokenInDataset
	bytesPerConversation := maxTurns * bytesPerTurn
	genPeakBytes := datasetCount * bytesPerConversation
	genPeakMiB := mib(float64(genPeakBytes))

	indexBytes := datasetCount * mmapIndexEntryBytes
	steadyVariable := mib(float64(indexBytes))

	peak := base + genPeakMiB
	return ComponentEstimate{Name: "DatasetManager", BaseMiB: base, VariableMiB: steadyVariable, PeakMiB: peak}
}

// estimateWorker estimates a single worker process: connection pool +
// in-flight requests + (multi-turn) session cache.
func estimateWorker(concurrencyPerWorker, avgOSL int, streaming bool, maxTurns, avgISL, connectionsPerWorker int) ComponentEstimate {
	base := float64(serviceBaseMiB["worker"] + pythonChildSubprocessBaseMiB)

	poolMiB := mib(float64(connectionsPerWorker * bytesPerConnection))
	perRequest := perRequestBytes(avgISL, avgOSL, streaming)
	inflightMiB := mib(float64(concurrencyPerWorker * perRequest))

	var sessionMiB float64
	if maxTurns > 1 {
		turnBytes := turnBaseBytes + avgISL*turnBytesPerToken
		extraTurnBytes := (maxTurns - 1) * turnBytes
		sessionMiB = mib(float64(concurrencyPerWorker * extraTurnBytes))
	}

	variable := poolMiB + inflightMiB + sessionMiB
	peak := base + variable*1.1
	return ComponentEstimate{Name: "Worker", BaseMiB: base, VariableMiB: variable, PeakMiB: peak}
}

// rpBufferMiB returns (rawBufferMiB, exportBufferMiB) for a single record
// processor.
func rpBufferMiB(avgOSL int, streaming bool) (float64, float64) {
	perToken := 4
	if streaming {
		perToken = sseBytesPerChunk
	}
	perRawRecordBytes := 1500 + avgOSL*perToken
	rawBufferMiB := mib(float64(rawBatchSize * perRawRecordBytes))
	exportBufferMiB := mib(float64(exportBatchSize * exportBytesPerRecord))
	return rawBufferMiB, exportBufferMiB
}

// estimateRecordProcessor estimates a single record processor: tokenizer cache
// + in-flight records + write buffers.
func estimateRecordProcessor(numModels, avgISL, avgOSL int, streaming bool, concurrencyPerRP int) ComponentEstimate {
	base := float64(serviceBaseMiB["record_processor"] + pythonChildSubprocessBaseMiB)

	tokenizerMiB := float64(numModels * tokenizerCacheMiB)
	perRecordBytes := perRequestBytes(avgISL, avgOSL, streaming)
	inflightMiB := mib(float64(concurrencyPerRP * perRecordBytes))
	rawBufferMiB, exportBufferMiB := rpBufferMiB(avgOSL, streaming)

	variable := tokenizerMiB + inflightMiB + rawBufferMiB + exportBufferMiB
	peak := base + variable*1.1
	return ComponentEstimate{Name: "RecordProcessor", BaseMiB: base, VariableMiB: variable, PeakMiB: peak}
}

// estimateGPUTelemetry estimates the GPU telemetry manager: columnar arrays
// per GPU per metric.
func estimateGPUTelemetry(numGPUs int, durationS, sampleIntervalS float64, numMetrics int, enabled bool) ComponentEstimate {
	if !enabled {
		return ComponentEstimate{Name: "GPU Telemetry", BaseMiB: 0, VariableMiB: 0, PeakMiB: 0}
	}
	base := float64(serviceBaseMiB["gpu_telemetry_manager"] + pythonSubprocessBaseMiB)

	interval := sampleIntervalS
	if interval < 0.1 {
		interval = 0.1
	}
	nSamples := int(durationS / interval)
	capacity := ceilPow2(nSamples)

	perGPUBytes := capacity*int64Bytes + numMetrics*capacity*float64Bytes
	totalBytes := float64(numGPUs*perGPUBytes) * growableArrayOverhead
	variable := mib(totalBytes)
	peak := base + variable
	return ComponentEstimate{Name: "GPU Telemetry", BaseMiB: base, VariableMiB: variable, PeakMiB: peak}
}

// estimateServerMetrics estimates the server metrics manager: scalar +
// histogram time series per endpoint.
func estimateServerMetrics(numEndpoints int, durationS, scrapeIntervalS float64, enabled bool, uniqueSeries, histogramCount, histogramBuckets int) ComponentEstimate {
	if !enabled {
		return ComponentEstimate{Name: "Server Metrics", BaseMiB: 0, VariableMiB: 0, PeakMiB: 0}
	}
	base := float64(serviceBaseMiB["server_metrics_manager"] + pythonSubprocessBaseMiB)

	interval := scrapeIntervalS
	if interval < 0.1 {
		interval = 0.1
	}
	nScrapes := int(durationS / interval)
	capacity := ceilPow2(nScrapes)

	scalarCount := uniqueSeries - histogramCount
	if scalarCount < 0 {
		scalarCount = 0
	}
	scalarBytes := scalarCount * capacity * (int64Bytes + float64Bytes)
	histBytes := histogramCount * capacity * (int64Bytes + 2*float64Bytes + histogramBuckets*float64Bytes)
	fetchBytes := nScrapes * int64Bytes * 2

	perEndpoint := float64(scalarBytes+histBytes+fetchBytes) * growableArrayOverhead
	totalBytes := float64(numEndpoints) * perEndpoint
	variable := mib(totalBytes)
	peak := base + variable
	return ComponentEstimate{Name: "Server Metrics", BaseMiB: base, VariableMiB: variable, PeakMiB: peak}
}

// estimateFixedService estimates a fixed-overhead service (subprocess +
// service baseline, no variable term).
func estimateFixedService(name string) ComponentEstimate {
	svc, ok := serviceBaseMiB[name]
	if !ok {
		svc = 20
	}
	base := float64(svc + pythonSubprocessBaseMiB)
	return ComponentEstimate{Name: name, BaseMiB: base, VariableMiB: 0, PeakMiB: base}
}

// aggregatePodComponent scales a per-process estimate to a pod total of
// count copies.
func aggregatePodComponent(name string, perUnit ComponentEstimate, count int) ComponentEstimate {
	return ComponentEstimate{
		Name:        name,
		BaseMiB:     perUnit.BaseMiB * float64(count),
		VariableMiB: perUnit.VariableMiB * float64(count),
		PeakMiB:     perUnit.PeakMiB * float64(count),
	}
}

// Params mirrors aiperf's MemoryEstimationParams: the workload shape that
// drives the host-side memory math.
type Params struct {
	TotalWorkers                 int
	WorkersPerPod                int
	NumWorkerPods                int
	RecordProcessorsPerPod       int
	MaxConcurrency               int
	TotalRequests                int
	TotalBenchmarkDurationS      float64
	DatasetCount                 int
	AvgISL                       int
	AvgOSL                       int
	MaxTurns                     int
	Streaming                    bool
	ListMetricBackend            string // "ragged" or "tdigest"
	ConnectionsPerWorker         int
	GPUTelemetryEnabled          bool
	ServerMetricsEnabled         bool
	NumGPUs                      int
	GPUSampleIntervalS           float64
	NumGPUMetrics                int
	NumServerMetricsEndpoints    int
	ServerMetricsScrapeIntervalS float64
	EstUniqueMetricSeries        int
	EstHistogramMetrics          int
	EstHistogramBuckets          int
	NumModels                    int
	NumStandardMetrics           int
}

// PodEstimate aggregates the component estimates for one pod type.
type PodEstimate struct {
	PodType    string
	Components []ComponentEstimate
	Replicas   int
}

// TotalSteadyStateMiB sums base+variable across the pod's components.
func (p PodEstimate) TotalSteadyStateMiB() float64 {
	total := 0.0
	for _, c := range p.Components {
		total += c.SteadyStateMiB()
	}
	return total
}

// TotalPeakMiB sums peak across the pod's components.
func (p PodEstimate) TotalPeakMiB() float64 {
	total := 0.0
	for _, c := range p.Components {
		total += c.PeakMiB
	}
	return total
}

// RecommendedRequestMiB is the steady-state total with the steady-state margin.
func (p PodEstimate) RecommendedRequestMiB() int {
	return int(math.Ceil(p.TotalSteadyStateMiB() * steadyStateMargin))
}

// RecommendedLimitMiB is the peak total with the peak margin.
func (p PodEstimate) RecommendedLimitMiB() int {
	return int(math.Ceil(p.TotalPeakMiB() * peakMargin))
}

// MemoryEstimate is the full cluster memory estimate.
type MemoryEstimate struct {
	Controller PodEstimate
	WorkerPod  PodEstimate
	Operator   PodEstimate
}

// TotalClusterMiB mirrors aiperf's total_cluster_mib (steady-state):
// controller + worker pod (x replicas) + operator.
func (e MemoryEstimate) TotalClusterMiB() float64 {
	return e.Controller.TotalSteadyStateMiB() +
		e.WorkerPod.TotalSteadyStateMiB()*float64(e.WorkerPod.Replicas) +
		e.Operator.TotalSteadyStateMiB()
}

// TotalClusterPeakMiB is the cluster-wide peak footprint.
func (e MemoryEstimate) TotalClusterPeakMiB() float64 {
	return e.Controller.TotalPeakMiB() +
		e.WorkerPod.TotalPeakMiB()*float64(e.WorkerPod.Replicas) +
		e.Operator.TotalPeakMiB()
}

func estimateController(p Params) PodEstimate {
	components := []ComponentEstimate{
		estimateFixedService("system_controller"),
		estimateFixedService("timing_manager"),
		estimateDatasetManager(p.DatasetCount, p.AvgISL, p.AvgOSL, p.MaxTurns),
		estimateRecordsManager(
			p.TotalRequests, p.NumStandardMetrics, p.AvgOSL,
			p.Streaming, p.ListMetricBackend, p.DatasetCount,
		),
		estimateFixedService("api_service"),
		estimateGPUTelemetry(
			p.NumGPUs, p.TotalBenchmarkDurationS, p.GPUSampleIntervalS,
			p.NumGPUMetrics, p.GPUTelemetryEnabled,
		),
		estimateServerMetrics(
			p.NumServerMetricsEndpoints, p.TotalBenchmarkDurationS,
			p.ServerMetricsScrapeIntervalS, p.ServerMetricsEnabled,
			p.EstUniqueMetricSeries, p.EstHistogramMetrics, p.EstHistogramBuckets,
		),
		estimateFixedService("results_sidecar"),
		ComponentEstimate{
			Name:        "ZMQ Proxies",
			BaseMiB:     float64(numZMQProxies * zmqProxyMiB),
			VariableMiB: 0,
			PeakMiB:     float64(numZMQProxies * zmqProxyMiB),
		},
	}
	return PodEstimate{PodType: "controller", Components: components, Replicas: 1}
}

func estimateWorkerPod(p Params) PodEstimate {
	concPerWorker := ceilDiv(p.MaxConcurrency, p.TotalWorkers)
	podConcurrency := ceilDiv(p.MaxConcurrency, p.NumWorkerPods)
	concPerRP := ceilDiv(podConcurrency, p.RecordProcessorsPerPod)
	rpQueue := rpQueueDepth(concPerRP, p.AvgISL, p.AvgOSL)

	wpm := estimateFixedService("worker_group_manager")
	worker := estimateWorker(
		concPerWorker, p.AvgOSL, p.Streaming, p.MaxTurns, p.AvgISL,
		p.ConnectionsPerWorker,
	)
	rp := estimateRecordProcessor(
		p.NumModels, p.AvgISL, p.AvgOSL, p.Streaming, rpQueue,
	)

	workersTotal := aggregatePodComponent(
		fmt.Sprintf("Workers (x%d)", p.WorkersPerPod), worker, p.WorkersPerPod,
	)
	rpTotal := aggregatePodComponent(
		fmt.Sprintf("RecordProcessors (x%d)", p.RecordProcessorsPerPod), rp,
		p.RecordProcessorsPerPod,
	)

	return PodEstimate{
		PodType:    "worker",
		Components: []ComponentEstimate{wpm, workersTotal, rpTotal},
		Replicas:   p.NumWorkerPods,
	}
}

func estimateOperator() PodEstimate {
	return PodEstimate{
		PodType: "operator",
		Components: []ComponentEstimate{
			{Name: "Operator", BaseMiB: 256, VariableMiB: 0, PeakMiB: 256},
		},
		Replicas: 1,
	}
}

// Estimate computes the cluster memory footprint for the given workload
// params, mirroring aiperf's MemoryEstimator.estimate().
func Estimate(p Params) MemoryEstimate {
	return MemoryEstimate{
		Controller: estimateController(p),
		WorkerPod:  estimateWorkerPod(p),
		Operator:   estimateOperator(),
	}
}
