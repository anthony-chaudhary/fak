package main

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

const (
	dispatchHostSpaceDefaultReserveBytes   int64 = 2 << 30
	dispatchHostSpaceDefaultPerWorkerBytes int64 = 1 << 30
	dispatchHostSpaceMaxBudgetBytes        int64 = 1 << 40
	dispatchHostSpaceMaxWorkers                  = 256
	dispatchHostSpaceRemediation                 = "fak worktree worker reap --all-cold"
)

type dispatchHostSpaceProbe func(string) (total, free int64, known bool)

type dispatchHostSpaceAdmission struct {
	OK                   bool   `json:"ok"`
	ReasonCode           string `json:"reason_code,omitempty"`
	Reason               string `json:"reason,omitempty"`
	TargetPath           string `json:"target_path"`
	FilesystemPath       string `json:"filesystem_path"`
	FilesystemTotalBytes int64  `json:"filesystem_total_bytes,omitempty"`
	FreeBytes            int64  `json:"free_bytes,omitempty"`
	ReserveBytes         int64  `json:"reserve_bytes"`
	PerWorkerBytes       int64  `json:"per_worker_bytes"`
	RequestedWorkers     int    `json:"requested_workers"`
	AdmittedWorkers      int    `json:"admitted_workers"`
	PredictedDemandBytes int64  `json:"predicted_demand_bytes"`
	AdmittedDemandBytes  int64  `json:"admitted_demand_bytes"`
	MeasurementKnown     bool   `json:"measurement_known"`
	ThresholdProvenance  string `json:"threshold_provenance"`
	Remediation          string `json:"remediation,omitempty"`
}

func dispatchHostSpaceExistingAncestor(path string) string {
	path = filepath.Clean(path)
	for {
		if _, err := os.Stat(path); err == nil {
			return path
		}
		parent := filepath.Dir(path)
		if parent == path {
			return path
		}
		path = parent
	}
}

func dispatchHostSpaceTarget() string {
	return workerworktree.DefaultRoot()
}

func dispatchHostSpaceAdmit(target string, requested int, reserve, perWorker int64, provenance string, probe dispatchHostSpaceProbe) dispatchHostSpaceAdmission {
	filesystemPath := dispatchHostSpaceExistingAncestor(target)
	receipt := dispatchHostSpaceAdmission{
		TargetPath:          target,
		FilesystemPath:      filesystemPath,
		ReserveBytes:        reserve,
		PerWorkerBytes:      perWorker,
		RequestedWorkers:    requested,
		ThresholdProvenance: strings.TrimSpace(provenance),
	}
	if receipt.ThresholdProvenance == "" {
		receipt.ThresholdProvenance = "dispatch-wave defaults"
	}
	if requested < 1 || requested > dispatchHostSpaceMaxWorkers || reserve < 0 || reserve > dispatchHostSpaceMaxBudgetBytes || perWorker <= 0 || perWorker > dispatchHostSpaceMaxBudgetBytes {
		receipt.ReasonCode = "INVALID_HOST_SPACE_BUDGET"
		receipt.Reason = fmt.Sprintf("workers must be 1..%d and byte budgets must be bounded", dispatchHostSpaceMaxWorkers)
		return receipt
	}
	if int64(requested) > math.MaxInt64/perWorker {
		receipt.ReasonCode = "INVALID_HOST_SPACE_BUDGET"
		receipt.Reason = "predicted worker demand overflows the host byte budget"
		return receipt
	}
	receipt.PredictedDemandBytes = int64(requested) * perWorker

	total, free, known := probe(filesystemPath)
	receipt.FilesystemTotalBytes = total
	receipt.FreeBytes = free
	receipt.MeasurementKnown = known
	if !known || free < 0 {
		receipt.ReasonCode = "HOST_FREE_SPACE_UNKNOWN"
		receipt.Reason = "managed-worktree filesystem free space is unknown; dispatch wave fails closed"
		receipt.Remediation = dispatchHostSpaceRemediation
		return receipt
	}

	usable := free - reserve
	if usable > 0 {
		capacity := usable / perWorker
		if capacity > int64(requested) {
			capacity = int64(requested)
		}
		receipt.AdmittedWorkers = int(capacity)
		receipt.AdmittedDemandBytes = capacity * perWorker
	}
	if receipt.AdmittedWorkers == 0 {
		receipt.ReasonCode = "HOST_FREE_SPACE_BUDGET"
		receipt.Reason = fmt.Sprintf("managed-worktree filesystem admits 0 of %d requested workers", requested)
		receipt.Remediation = dispatchHostSpaceRemediation
		return receipt
	}
	if receipt.AdmittedWorkers < requested {
		receipt.ReasonCode = "HOST_FREE_SPACE_PARTIAL"
		receipt.Reason = fmt.Sprintf("managed-worktree filesystem admits %d of %d requested workers", receipt.AdmittedWorkers, requested)
		receipt.Remediation = dispatchHostSpaceRemediation
	}
	receipt.OK = true
	return receipt
}

func dispatchHostSpaceDefaultProbe(path string) (int64, int64, bool) {
	return compute.DiskInfo(path)
}

var dispatchHostSpaceProbeFn dispatchHostSpaceProbe = dispatchHostSpaceDefaultProbe
