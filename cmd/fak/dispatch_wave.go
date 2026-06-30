package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

func runDispatchWave(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("dispatch wave", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: current directory)")
	count := fs.Int("count", 2, "number of distinct account pools to allocate")
	maxWorkers := fs.Int("max-workers", dispatchtick.DefaultMaxWorkers, "hard cap on live workers, enforced by each tick's preflight")
	backend := fs.String("backend", "claude", "worker backend (claude|opencode|codex)")
	workKind := fs.String("work-kind", "", "switcher work kind (default follows --backend)")
	lane := fs.String("lane", "", "pin every tick to this repo lane (default: busiest-lane pick)")
	excludeLane := fs.String("exclude-lane", "", "comma-separated lanes to drop from the busiest-pick")
	settleS := fs.Float64("settle-s", 2.0, "seconds to wait after each live spawn")
	noLedger := fs.Bool("no-loop-ledger", false, "disable loop-ledger append for spawned ticks")
	live := fs.Bool("live", false, "actually spawn workers")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	root := *workspace
	if root == "" {
		wd, err := os.Getwd()
		if err != nil {
			fmt.Fprintf(stderr, "fak dispatch wave: getwd: %v\n", err)
			return 1
		}
		root = wd
	}
	backendNorm, err := dispatchtick.NormalizeBackend(*backend)
	if err != nil {
		fmt.Fprintf(stderr, "fak dispatch wave: %v\n", err)
		return 2
	}
	wk := strings.TrimSpace(*workKind)
	if wk == "" {
		wk = dispatchtick.DefaultWorkKind(backendNorm)
	}
	if *count <= 0 {
		fmt.Fprintln(stderr, "fak dispatch wave: --count must be > 0")
		return 2
	}

	rows, err := dispatchReadAccountRoster(root)
	if err != nil {
		fmt.Fprintf(stderr, "fak dispatch wave: allocate accounts: %v\n", err)
		return 1
	}
	alloc := dispatchtick.AllocateWave(dispatchtick.AccountWaveInput{
		Rows:     rows,
		Count:    *count,
		WorkKind: wk,
		Product:  dispatchtick.ProductForBackend(backendNorm),
	})
	lanes := alloc.Lanes
	waveID := alloc.WaveID
	shortfall := alloc.Shortfall
	rec := map[string]any{
		"schema":      "fleet-issue-dispatch-wave/1",
		"workspace":   root,
		"live":        *live,
		"backend":     backendNorm,
		"work_kind":   wk,
		"requested":   *count,
		"granted":     len(lanes),
		"shortfall":   shortfall,
		"wave_id":     waveID,
		"allocation":  scrubDispatchSecrets(alloc.Map()),
		"ticks":       []any{},
		"spawned":     0,
		"stop_reason": "",
		"ok":          false,
	}
	if len(lanes) == 0 {
		rec["stop_reason"] = firstString(alloc.Reason, "no distinct account pools available")
		return writeDispatchWaveResult(stdout, stderr, rec, *asJSON)
	}

	ticks := []any{}
	spawned := 0
	limit := minInt(*count, len(lanes))
	if !*live {
		limit = 1
	}
	for i := 0; i < limit; i++ {
		acct := accountFromWaveLane(lanes[i])
		mem := dispatchtick.Membership{Rank: i, WaveID: waveID, Size: len(lanes), Shortfall: shortfall}
		payload, err := evaluateDispatchTick(dispatchTickOptions{
			Workspace:      root,
			MaxWorkers:     *maxWorkers,
			WorkKind:       wk,
			Lane:           *lane,
			Backend:        backendNorm,
			ExcludeLanes:   dispatchSplitCSV(*excludeLane),
			Live:           *live,
			Refresh:        i == 0,
			CooldownMin:    dispatchtick.DefaultCooldownMinutes,
			WorkerTimeoutS: dispatchtick.DefaultWorkerTimeoutS,
			SpawnProbeS:    dispatchtick.DefaultSpawnProbeS,
			RecordLoop:     !*noLedger,
			Account:        &acct,
			Membership:     &mem,
		}, stderr)
		if err != nil {
			ticks = append(ticks, map[string]any{"ok": false, "error": err.Error(), "rank": i})
			rec["stop_reason"] = err.Error()
			break
		}
		payload["wave_rank"] = i
		ticks = append(ticks, payload)
		if dispatchMapString(payload, "action") == "spawned" {
			spawned++
			if *settleS > 0 {
				time.Sleep(time.Duration(*settleS * float64(time.Second)))
			}
			continue
		}
		if !*live {
			rec["stop_reason"] = "dry-run: planned the first wave tick only; re-run with --live to spawn"
		} else {
			rec["stop_reason"] = firstString(dispatchMapString(payload, "verdict"), dispatchMapString(payload, "action"))
		}
		break
	}
	rec["ticks"] = ticks
	rec["spawned"] = spawned
	if rec["stop_reason"] == "" {
		rec["stop_reason"] = "filled requested wave"
	}
	rec["ok"] = !*live || spawned > 0 || len(ticks) > 0 && dispatchMapBool(ticks[len(ticks)-1].(map[string]any), "ok")
	return writeDispatchWaveResult(stdout, stderr, rec, *asJSON)
}

func writeDispatchWaveResult(stdout, stderr io.Writer, rec map[string]any, asJSON bool) int {
	if asJSON {
		if err := writeIndentedJSON(stdout, rec); err != nil {
			fmt.Fprintf(stderr, "fak dispatch wave: encode json: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprint(stdout, renderDispatchWave(rec))
	}
	if dispatchMapBool(rec, "ok") {
		return 0
	}
	return 1
}

func renderDispatchWave(rec map[string]any) string {
	var b strings.Builder
	mode := "dry-run"
	if dispatchMapBool(rec, "live") {
		mode = "live"
	}
	fmt.Fprintf(&b, "issue-dispatch-wave: %s  requested=%d granted=%d spawned=%d backend=%s\n",
		mode, dispatchMapInt(rec, "requested"), dispatchMapInt(rec, "granted"),
		dispatchMapInt(rec, "spawned"), dispatchMapString(rec, "backend"))
	if id := dispatchMapString(rec, "wave_id"); id != "" {
		fmt.Fprintf(&b, "  wave_id: %s\n", id)
	}
	if reason := dispatchMapString(rec, "stop_reason"); reason != "" {
		fmt.Fprintf(&b, "  stop: %s\n", reason)
	}
	if !dispatchMapBool(rec, "live") {
		fmt.Fprintln(&b, "  (dry-run - re-run with --live to spawn the wave)")
	}
	return b.String()
}

func waveAccountLanes(doc map[string]any) []map[string]any {
	raw, _ := doc["lanes"].([]any)
	out := make([]map[string]any, 0, len(raw))
	for _, item := range raw {
		if m, ok := item.(map[string]any); ok {
			out = append(out, m)
		}
	}
	return out
}

func accountFromWaveLane(m dispatchtick.AccountWaveLane) dispatchtick.Account {
	return dispatchtick.Account{
		Tag:   firstString(m.Tag, m.Account),
		Tier:  m.SelectedTier,
		Model: m.Model,
		Dir:   m.ConfigDir,
	}
}

func firstAny(vals ...any) any {
	for _, v := range vals {
		if v != nil {
			return v
		}
	}
	return nil
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func scrubDispatchSecrets(v any) any {
	switch x := v.(type) {
	case map[string]any:
		out := make(map[string]any, len(x))
		for k, val := range x {
			if dispatchSecretKey(k) {
				continue
			}
			out[k] = scrubDispatchSecrets(val)
		}
		return out
	case []any:
		out := make([]any, len(x))
		for i, val := range x {
			out[i] = scrubDispatchSecrets(val)
		}
		return out
	default:
		return v
	}
}

func dispatchSecretKey(k string) bool {
	k = strings.ToLower(strings.TrimSpace(k))
	return strings.Contains(k, "token") || strings.Contains(k, "secret") || strings.Contains(k, "api_key") || strings.Contains(k, "apikey")
}
