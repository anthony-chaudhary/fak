// Package localappmetrics joins operation events to outcomes without content or stable device identity.
package localappmetrics

import (
	"errors"
	"sort"
	"time"
)

type Operation struct {
	JoinID                                                                string
	InstallStarted, ReadyAt                                               time.Time
	Eligible, LocalAttempted, LocalAccepted, ResultUsed                   bool
	HandoffReason                                                         string
	TTFT, EndToEnd                                                        time.Duration
	PeakMemoryBytes, PeakDiskBytes                                        int64
	ForegroundImpact                                                      time.Duration
	Crash, UpdateRollback                                                 bool
	LocalCost, RetryCost, VerificationCost, RemoteCost, CloudBaselineCost float64
}
type Report struct {
	Operations          int            `json:"operations"`
	InstallReadySuccess int            `json:"install_ready_success"`
	InstallReadyP50     time.Duration  `json:"install_ready_p50"`
	InstallReadyP95     time.Duration  `json:"install_ready_p95"`
	Eligible            int            `json:"eligible"`
	LocalAttempted      int            `json:"local_attempted"`
	LocalAccepted       int            `json:"local_accepted"`
	ResultUsed          int            `json:"result_used"`
	Handoffs            map[string]int `json:"handoffs"`
	TTFTP50             time.Duration  `json:"ttft_p50"`
	TTFTP95             time.Duration  `json:"ttft_p95"`
	EndToEndP50         time.Duration  `json:"end_to_end_p50"`
	EndToEndP95         time.Duration  `json:"end_to_end_p95"`
	PeakMemoryBytes     int64          `json:"peak_memory_bytes"`
	PeakDiskBytes       int64          `json:"peak_disk_bytes"`
	ForegroundImpactP95 time.Duration  `json:"foreground_impact_p95"`
	CrashFree           int            `json:"crash_free"`
	UpdateRollbacks     int            `json:"update_rollbacks"`
	NetCloudSavings     float64        `json:"net_cloud_savings"`
}

var ErrSensitive = errors.New("localappmetrics: stable identity or content is not accepted")

func Aggregate(rows []Operation) (Report, error) {
	r := Report{Operations: len(rows), Handoffs: map[string]int{}}
	seen := map[string]bool{}
	var installs, ttft, e2e, foreground []time.Duration
	for _, o := range rows {
		if o.JoinID == "" || seen[o.JoinID] {
			return Report{}, ErrSensitive
		}
		seen[o.JoinID] = true
		if o.Eligible {
			r.Eligible++
		}
		if o.LocalAttempted {
			r.LocalAttempted++
		}
		if o.LocalAccepted {
			r.LocalAccepted++
		}
		if o.ResultUsed {
			r.ResultUsed++
		}
		if !o.InstallStarted.IsZero() && !o.ReadyAt.IsZero() && !o.ReadyAt.Before(o.InstallStarted) {
			r.InstallReadySuccess++
			installs = append(installs, o.ReadyAt.Sub(o.InstallStarted))
		}
		if o.HandoffReason != "" {
			r.Handoffs[o.HandoffReason]++
		}
		if o.TTFT > 0 {
			ttft = append(ttft, o.TTFT)
		}
		if o.EndToEnd > 0 {
			e2e = append(e2e, o.EndToEnd)
		}
		if o.ForegroundImpact > 0 {
			foreground = append(foreground, o.ForegroundImpact)
		}
		if o.PeakMemoryBytes > r.PeakMemoryBytes {
			r.PeakMemoryBytes = o.PeakMemoryBytes
		}
		if o.PeakDiskBytes > r.PeakDiskBytes {
			r.PeakDiskBytes = o.PeakDiskBytes
		}
		if !o.Crash {
			r.CrashFree++
		}
		if o.UpdateRollback {
			r.UpdateRollbacks++
		}
		if o.LocalAccepted && o.ResultUsed && o.RemoteCost == 0 {
			saving := o.CloudBaselineCost - o.LocalCost - o.RetryCost - o.VerificationCost - o.RemoteCost
			if saving > 0 {
				r.NetCloudSavings += saving
			}
		}
	}
	r.InstallReadyP50, r.InstallReadyP95 = quantiles(installs)
	r.TTFTP50, r.TTFTP95 = quantiles(ttft)
	r.EndToEndP50, r.EndToEndP95 = quantiles(e2e)
	_, r.ForegroundImpactP95 = quantiles(foreground)
	return r, nil
}
func quantiles(v []time.Duration) (time.Duration, time.Duration) {
	if len(v) == 0 {
		return 0, 0
	}
	sort.Slice(v, func(i, j int) bool { return v[i] < v[j] })
	return v[(len(v)-1)/2], v[((len(v)-1)*95+99)/100]
}
