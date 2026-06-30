package main

import (
	"crypto/sha256"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchorder"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

const dispatchPriceSchema = "fak-dispatch-price/1"

type dispatchPriceInput struct {
	Agents []dispatchPriceAgent `json:"agents"`
}

type dispatchPriceAgent struct {
	Name    string   `json:"name"`
	Lane    string   `json:"lane,omitempty"`
	LeaseID string   `json:"lease_id,omitempty"`
	Tree    []string `json:"tree,omitempty"`
	Mode    string   `json:"mode,omitempty"`
}

type dispatchPriceReport struct {
	Schema               string                            `json:"schema"`
	OK                   bool                              `json:"ok"`
	Action               string                            `json:"action"`
	ActionReason         string                            `json:"action_reason"`
	Log                  string                            `json:"log"`
	PlanID               string                            `json:"plan_id,omitempty"`
	Requested            int                               `json:"requested"`
	SafeConcurrency      int                               `json:"safe_concurrency"`
	SafeNow              []string                          `json:"safe_now"`
	WaveCount            int                               `json:"wave_count"`
	Waves                []dispatchPriceWave               `json:"waves"`
	LaunchPlan           []dispatchLaunchWave              `json:"launch_plan"`
	LaneSerialWaveCount  int                               `json:"lane_serial_wave_count"`
	ScopedParallelGain   int                               `json:"scoped_parallelism_gain"`
	CollisionWavePenalty int                               `json:"collision_wave_penalty"`
	SafeConcurrencyPct   int                               `json:"safe_concurrency_pct"`
	SameLaneParallelism  int                               `json:"same_lane_parallelism"`
	Collisions           []dispatchorder.Collision         `json:"collisions,omitempty"`
	Repartition          []dispatchorder.RepartitionAdvice `json:"repartition,omitempty"`
	CollisionsAvoided    int                               `json:"collisions_avoided"`
	SerializationWasted  int                               `json:"serialization_wasted"`
	ExpectedRework       int                               `json:"expected_rework"`
	Candidates           []dispatchPriceCandidate          `json:"candidates"`
}

type dispatchPriceWave struct {
	Index  int      `json:"index"`
	Agents []string `json:"agents"`
	Size   int      `json:"size"`
}

type dispatchLaunchWave struct {
	Index   int                    `json:"index"`
	Size    int                    `json:"size"`
	Targets []dispatchLaunchTarget `json:"targets"`
}

type dispatchLaunchTarget struct {
	ID          string                    `json:"id"`
	Lane        string                    `json:"lane,omitempty"`
	LeaseID     string                    `json:"lease_id"`
	Issue       int                       `json:"issue,omitempty"`
	Tree        []string                  `json:"tree,omitempty"`
	Mode        string                    `json:"mode,omitempty"`
	Scoped      bool                      `json:"scoped"`
	ScopeSource string                    `json:"scope_source,omitempty"`
	Disposition dispatchorder.Disposition `json:"disposition"`
	Reason      string                    `json:"reason,omitempty"`
	TickArgs    []string                  `json:"dispatch_tick_args,omitempty"`
}

type dispatchPriceCandidate struct {
	Name         string                    `json:"name"`
	Lane         string                    `json:"lane,omitempty"`
	LeaseID      string                    `json:"lease_id"`
	Tree         []string                  `json:"tree,omitempty"`
	Mode         string                    `json:"mode,omitempty"`
	TreeSource   string                    `json:"tree_source"`
	Disposition  dispatchorder.Disposition `json:"disposition"`
	Reason       string                    `json:"reason"`
	CollidesWith []string                  `json:"collides_with,omitempty"`
	Rank         int                       `json:"rank"`
}

func runDispatchPrice(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("dispatch price", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root for lane-tree resolution (default: current directory)")
	inPath := fs.String("in", "", "read proposed fan-out JSON from this file (default: stdin)")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak dispatch price: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	root := strings.TrimSpace(*workspace)
	if root == "" {
		wd, err := os.Getwd()
		if err != nil {
			fmt.Fprintf(stderr, "fak dispatch price: getwd: %v\n", err)
			return 1
		}
		root = wd
	}
	raw, code := readDispatchInput(stderr, *inPath)
	if code != 0 {
		return code
	}
	agents, err := parseDispatchPriceAgents(raw)
	if err != nil {
		fmt.Fprintf(stderr, "fak dispatch price: parse input: %v\n", err)
		return 1
	}
	taxonomy, err := dispatchLoadLaneTaxonomy(root)
	if err != nil {
		fmt.Fprintf(stderr, "fak dispatch price: lane taxonomy: %v\n", err)
		return 1
	}
	rep := buildDispatchPriceReport(agents, taxonomy)
	if *asJSON {
		if err := writeIndentedJSONNoEscape(stdout, rep); err != nil {
			fmt.Fprintf(stderr, "fak dispatch price: encode json: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprint(stdout, renderDispatchPrice(rep))
	}
	return 0
}

func parseDispatchPriceAgents(raw []byte) ([]dispatchPriceAgent, error) {
	var obj dispatchPriceInput
	if err := json.Unmarshal(raw, &obj); err == nil && obj.Agents != nil {
		return normalizeDispatchPriceAgents(obj.Agents), nil
	}
	var arr []dispatchPriceAgent
	if err := json.Unmarshal(raw, &arr); err != nil {
		return nil, err
	}
	return normalizeDispatchPriceAgents(arr), nil
}

func normalizeDispatchPriceAgents(agents []dispatchPriceAgent) []dispatchPriceAgent {
	out := make([]dispatchPriceAgent, 0, len(agents))
	for i, agent := range agents {
		agent.Name = strings.TrimSpace(agent.Name)
		if agent.Name == "" {
			agent.Name = fmt.Sprintf("agent-%d", i+1)
		}
		agent.Lane = strings.TrimSpace(agent.Lane)
		agent.LeaseID = strings.TrimSpace(agent.LeaseID)
		agent.Mode = strings.TrimSpace(agent.Mode)
		out = append(out, agent)
	}
	return out
}

func buildDispatchPriceReport(agents []dispatchPriceAgent, taxonomy dispatchtick.LaneTaxonomy) dispatchPriceReport {
	cands := make([]dispatchorder.Candidate, 0, len(agents))
	meta := make(map[string]dispatchPriceCandidate, len(agents))
	seen := map[string]int{}
	for i, agent := range agents {
		id := uniqueDispatchPriceID(agent.Name, seen)
		tree, source := dispatchPriceTree(agent, taxonomy)
		leaseID := dispatchPriceLeaseID(agent, id, source)
		mode := agent.Mode
		if strings.TrimSpace(mode) == "" {
			mode = "exclusive"
		}
		meta[id] = dispatchPriceCandidate{
			Name:       id,
			Lane:       agent.Lane,
			LeaseID:    leaseID,
			Tree:       append([]string(nil), tree...),
			Mode:       mode,
			TreeSource: source,
		}
		cands = append(cands, dispatchorder.Candidate{
			ID:          id,
			Key:         id,
			Lane:        leaseID,
			Tree:        tree,
			Mode:        mode,
			CreatedUnix: int64(len(agents) - i),
			UpdatedUnix: int64(len(agents) - i),
		})
	}

	res := dispatchorder.Plan(dispatchorder.Input{
		Candidates:      cands,
		NowUnix:         time.Unix(0, 0).Unix(),
		CooldownSeconds: -1,
	})
	candidates := make([]dispatchPriceCandidate, 0, len(res.Order))
	for _, row := range res.Order {
		cand := meta[row.ID]
		cand.Disposition = row.Disposition
		cand.Reason = row.Reason
		cand.CollidesWith = append([]string(nil), row.CollidesWith...)
		cand.Rank = row.Rank
		candidates = append(candidates, cand)
	}
	action, reason := dispatchWaveAction(len(cands), res.KeepCount, res.CollisionsAvoided, res.SerializationWasted)
	waves := dispatchPriceWaves(candidates, res.Collisions, res.Keep)
	laneSerialWaves := dispatchPriceLaneSerialWaveCount(candidates)
	launchPlan := dispatchPriceLaunchPlan(waves, candidates)
	return dispatchPriceReport{
		Schema:               dispatchPriceSchema,
		OK:                   true,
		Action:               action,
		ActionReason:         reason,
		Log:                  "predictive price uses fak dispatchorder prefix geometry; acquire-time lane leases still must run as the reactive floor",
		PlanID:               dispatchLaunchPlanID(launchPlan),
		Requested:            len(cands),
		SafeConcurrency:      res.SafeConcurrency,
		SafeNow:              append([]string(nil), res.Keep...),
		WaveCount:            len(waves),
		Waves:                waves,
		LaunchPlan:           launchPlan,
		LaneSerialWaveCount:  laneSerialWaves,
		ScopedParallelGain:   positiveDelta(laneSerialWaves, len(waves)),
		CollisionWavePenalty: positiveDelta(len(waves), laneSerialWaves),
		SafeConcurrencyPct:   dispatchWavePct(res.SafeConcurrency, len(cands)),
		SameLaneParallelism:  dispatchPriceSameLaneParallelism(candidates),
		Collisions:           res.Collisions,
		Repartition:          res.Repartition,
		CollisionsAvoided:    res.CollisionsAvoided,
		SerializationWasted:  res.SerializationWasted,
		ExpectedRework:       res.CollisionsAvoided + res.SerializationWasted,
		Candidates:           candidates,
	}
}

func dispatchLaunchPlanID(plan []dispatchLaunchWave) string {
	if len(plan) == 0 {
		return ""
	}
	return dispatchStablePlanID(plan)
}

func dispatchStablePlanID(plan any) string {
	raw, err := json.Marshal(plan)
	if err != nil || len(raw) == 0 || string(raw) == "null" {
		return ""
	}
	sum := sha256.Sum256(raw)
	return fmt.Sprintf("plan-%x", sum[:8])
}

func dispatchPriceLaunchPlan(waves []dispatchPriceWave, candidates []dispatchPriceCandidate) []dispatchLaunchWave {
	if len(waves) == 0 {
		return nil
	}
	byName := map[string]dispatchPriceCandidate{}
	for _, cand := range candidates {
		byName[cand.Name] = cand
	}
	return dispatchLaunchPlanFromWaves(waves, func(id string) dispatchLaunchTarget {
		cand, ok := byName[id]
		if !ok {
			return dispatchLaunchTarget{ID: id}
		}
		return dispatchPriceLaunchTarget(cand)
	})
}

func dispatchLaunchPlanFromWaves(waves []dispatchPriceWave, targetFor func(string) dispatchLaunchTarget) []dispatchLaunchWave {
	out := make([]dispatchLaunchWave, 0, len(waves))
	for _, wave := range waves {
		targets := make([]dispatchLaunchTarget, 0, len(wave.Agents))
		for _, id := range wave.Agents {
			targets = append(targets, targetFor(id))
		}
		out = append(out, dispatchLaunchWave{Index: wave.Index, Size: len(targets), Targets: targets})
	}
	return out
}

func dispatchPriceLaunchTarget(cand dispatchPriceCandidate) dispatchLaunchTarget {
	mode := strings.TrimSpace(cand.Mode)
	if mode == "" {
		mode = "exclusive"
	}
	return dispatchLaunchTarget{
		ID:          cand.Name,
		Lane:        cand.Lane,
		LeaseID:     cand.LeaseID,
		Tree:        append([]string(nil), cand.Tree...),
		Mode:        mode,
		Scoped:      cand.TreeSource == "input" && len(cand.Tree) > 0,
		ScopeSource: cand.TreeSource,
		Disposition: cand.Disposition,
		Reason:      cand.Reason,
	}
}

func dispatchPriceLaneSerialWaveCount(candidates []dispatchPriceCandidate) int {
	if len(candidates) == 0 {
		return 0
	}
	keys := make([]string, 0, len(candidates))
	for _, cand := range candidates {
		key := strings.TrimSpace(cand.Lane)
		if key == "" {
			key = strings.TrimSpace(cand.LeaseID)
		}
		if key == "" {
			key = cand.Name
		}
		keys = append(keys, key)
	}
	return dispatchLaneSerialWaveCount(keys)
}

func dispatchLaneSerialWaveCount(keys []string) int {
	byLane := map[string]int{}
	for _, key := range keys {
		byLane[key]++
	}
	max := 0
	for _, count := range byLane {
		if count > max {
			max = count
		}
	}
	return max
}

func positiveDelta(a, b int) int {
	if a > b {
		return a - b
	}
	return 0
}

func dispatchPriceWaves(candidates []dispatchPriceCandidate, collisions []dispatchorder.Collision, safeNow []string) []dispatchPriceWave {
	if len(candidates) == 0 {
		return nil
	}
	ids := make([]string, 0, len(candidates))
	for _, cand := range candidates {
		ids = append(ids, cand.Name)
	}
	return dispatchWavesForIDs(ids, collisions, safeNow)
}

func dispatchWavesForIDs(ids []string, collisions []dispatchorder.Collision, safeNow []string) []dispatchPriceWave {
	collides := map[string]map[string]bool{}
	for _, c := range collisions {
		if collides[c.A] == nil {
			collides[c.A] = map[string]bool{}
		}
		if collides[c.B] == nil {
			collides[c.B] = map[string]bool{}
		}
		collides[c.A][c.B] = true
		collides[c.B][c.A] = true
	}
	remaining := map[string]bool{}
	for _, id := range ids {
		remaining[id] = true
	}
	var waves []dispatchPriceWave
	first := append([]string(nil), safeNow...)
	if len(first) > 0 {
		for _, id := range first {
			delete(remaining, id)
		}
		waves = append(waves, dispatchPriceWave{Index: 1, Agents: first, Size: len(first)})
	}
	for len(remaining) > 0 {
		var wave []string
		for _, id := range ids {
			if !remaining[id] {
				continue
			}
			ok := true
			for _, picked := range wave {
				if collides[id][picked] {
					ok = false
					break
				}
			}
			if ok {
				wave = append(wave, id)
			}
		}
		if len(wave) == 0 {
			for _, id := range ids {
				if remaining[id] {
					wave = append(wave, id)
					break
				}
			}
		}
		for _, id := range wave {
			delete(remaining, id)
		}
		waves = append(waves, dispatchPriceWave{Index: len(waves) + 1, Agents: wave, Size: len(wave)})
	}
	return waves
}

func dispatchPriceTree(agent dispatchPriceAgent, taxonomy dispatchtick.LaneTaxonomy) ([]string, string) {
	if len(agent.Tree) > 0 {
		return append([]string(nil), agent.Tree...), "input"
	}
	if agent.Lane != "" {
		if tree := taxonomy.Trees[agent.Lane]; len(tree) > 0 {
			return append([]string(nil), tree...), "lane"
		}
	}
	return nil, "unknown"
}

func dispatchPriceLeaseID(agent dispatchPriceAgent, id, source string) string {
	if agent.LeaseID != "" {
		return agent.LeaseID
	}
	if source == "lane" && agent.Lane != "" {
		return "price-" + cleanDispatchLeaseToken(agent.Lane)
	}
	return "price-" + cleanDispatchLeaseToken(id)
}

func uniqueDispatchPriceID(name string, seen map[string]int) string {
	n := seen[name]
	seen[name] = n + 1
	if n == 0 {
		return name
	}
	return fmt.Sprintf("%s-%d", name, n+1)
}

func dispatchPriceSameLaneParallelism(candidates []dispatchPriceCandidate) int {
	byLane := map[string]int{}
	for _, cand := range candidates {
		if cand.Disposition != dispatchorder.DispKeep || strings.TrimSpace(cand.Lane) == "" {
			continue
		}
		byLane[cand.Lane]++
	}
	extra := 0
	for _, n := range byLane {
		if n > 1 {
			extra += n - 1
		}
	}
	return extra
}

func renderDispatchPrice(rep dispatchPriceReport) string {
	var b strings.Builder
	planID := ""
	if rep.PlanID != "" {
		planID = " plan_id=" + rep.PlanID
	}
	fmt.Fprintf(&b, "dispatch price:%s %s requested=%d safe=%d waves=%d lane_serial=%d scoped_gain=%d collision_penalty=%d collisions=%d expected_rework=%d\n",
		planID, rep.Action, rep.Requested, rep.SafeConcurrency, rep.WaveCount, rep.LaneSerialWaveCount, rep.ScopedParallelGain, rep.CollisionWavePenalty, rep.CollisionsAvoided, rep.ExpectedRework)
	if len(rep.SafeNow) > 0 {
		fmt.Fprintf(&b, "  safe_now: %s\n", strings.Join(rep.SafeNow, ","))
	}
	if len(rep.Waves) > 0 {
		fmt.Fprintf(&b, "  schedule: %s\n", renderDispatchPriceWaves(rep.Waves))
	}
	if rep.SameLaneParallelism > 0 {
		fmt.Fprintf(&b, "  same_lane_parallelism=%d\n", rep.SameLaneParallelism)
	}
	for _, cand := range rep.Candidates {
		if cand.Disposition != dispatchorder.DispCollisionRisk {
			continue
		}
		fmt.Fprintf(&b, "  collision %-16s reason=%s peers=%s\n",
			cand.Name, cand.Reason, strings.Join(cand.CollidesWith, ","))
	}
	for _, adv := range rep.Repartition {
		fmt.Fprintf(&b, "  repartition %-16s action=%s peers=%s\n",
			adv.Candidate, adv.Action, strings.Join(adv.CollidesWith, ","))
	}
	fmt.Fprintf(&b, "Action: %s\n", rep.Action)
	return b.String()
}

func renderDispatchPriceWaves(waves []dispatchPriceWave) string {
	parts := make([]string, 0, len(waves))
	for _, wave := range waves {
		parts = append(parts, fmt.Sprintf("wave%d[%s]", wave.Index, strings.Join(wave.Agents, ",")))
	}
	return strings.Join(parts, " -> ")
}
