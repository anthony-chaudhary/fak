package frontierswe

import (
	"encoding/json"
	"regexp"
	"strconv"
	"strings"
)

// This file is the Go port of the upstream FrontierSWE reference scorer,
// scripts/score_from_reward.py (the published oracle). It turns a trial's
// reward.json into the leaderboard number in two steps, field-for-field with
// the Python:
//
//	ExtractScore(reward, task) -> (correctness in [0,1], speedup or nil)
//	GatedScore(correctness, speedup, task) -> leaderboard score
//
// The category gates, the libexpat-to-x86asm per-module partial-correctness
// weights, the notebook-compression rule, and frogsgame-rl's board-count
// normalization all mirror the Python verbatim. The anti-cheat zeroing — a trial
// flagged in scoring/anticheat.json scores 0 — is layered on top via Score /
// GatedScoreAntiCheat, because the reference script's own gate signature does
// not carry the anti-cheat flag (the upstream pipeline applies it separately).
// See docs/benchmarks/FRONTIERSWE-SCORING-PARITY.md for the runtime-vs-oracle
// mapping and known discrepancies.

// defaultSSIMThreshold is the revideo-perf-opt SSIM gate. The reference script's
// extract_score default is 0.99; the CLI default is 0.95. ExtractScore uses the
// extract_score default (0.99) so the pure port matches the library function;
// ExtractScoreSSIM lets a caller pass the CLI's 0.95.
const defaultSSIMThreshold = 0.99

// libexpatModuleWeights is the per-module partial-correctness weight table for
// libexpat-to-x86asm, mirroring _LIBEXPAT_MODULE_WEIGHTS in the reference
// script. acc_tests has weight 0 (excluded from the weighted reconstruction).
var libexpatModuleWeights = map[string]int{
	"smoke_tier":    1,
	"basic_tests":   3,
	"ns_tests":      2,
	"misc_tests":    1,
	"alloc_tests":   2,
	"nsalloc_tests": 1,
	"acc_tests":     0,
}

// libexpatModuleOrder is the deterministic iteration order over the weight
// table. Go map iteration is unordered; the reconstruction is a sum so order
// does not change the result, but a fixed order keeps the port auditable.
var libexpatModuleOrder = []string{
	"smoke_tier", "basic_tests", "ns_tests", "misc_tests",
	"alloc_tests", "nsalloc_tests", "acc_tests",
}

// Reward is the typed view of a trial's reward.json. FrontierSWE reward files
// are heterogeneous across tasks — some carry a flat "reward", others nest
// correctness under additional_data, others list subscores — so the fields here
// are the union the reference scorer reads, all optional. A field absent from
// the JSON stays at its zero value, which is exactly how the Python's
// reward_data.get(...) defaults behave.
type Reward struct {
	Reward             *float64        `json:"reward"`
	Score              *float64        `json:"score"`
	Reason             string          `json:"reason"`
	AdditionalData     *AdditionalData `json:"additional_data"`
	Subscores          []Subscore      `json:"subscores"`
	CorrectnessOK      bool            `json:"correctness_ok"`
	GeometricMean      *float64        `json:"geometric_mean_speedup"`
	CorrectnessDetails []SSIMDetail    `json:"correctness_details"`
	HardFailReasons    []string        `json:"hard_fail_reasons"`
	ParityPassRate     *float64        `json:"parity_pass_rate"`
}

// AdditionalData mirrors reward.json's additional_data object — the per-task
// correctness/perf detail the reference scorer reaches into. The
// "correctness_passed" key is polymorphic across tasks (an int count for
// ffmpeg-swscale-rewrite, a bare bool for cranelift-codegen-opt), so it is
// decoded by a custom UnmarshalJSON into both CorrectnessPassedN and
// CorrectnessBool.
type AdditionalData struct {
	TestsPassed        *int              `json:"tests_passed"`
	TestsTotal         *int              `json:"tests_total"`
	Modules            map[string]Module `json:"modules"`
	CorrectnessPassedN *int              `json:"-"`
	CorrectnessBool    *bool             `json:"-"`
	CorrectnessTotal   *int              `json:"correctness_total"`
	GeometricMean      *float64          `json:"geometric_mean_speedup"`
	GeoMeanSpeedup     *float64          `json:"geo_mean_speedup"`
	WeightedHarmonic   *float64          `json:"weighted_harmonic_mean"`
	Correctness        *Correctness      `json:"correctness"`
}

// additionalDataAlias is AdditionalData without the custom unmarshaler, used to
// decode the non-polymorphic fields without infinite recursion.
type additionalDataAlias AdditionalData

// UnmarshalJSON decodes additional_data, resolving the polymorphic
// "correctness_passed" key (int for ffmpeg, bool for cranelift) into the two
// typed fields. A numeric value populates CorrectnessPassedN; a boolean value
// populates CorrectnessBool.
func (a *AdditionalData) UnmarshalJSON(b []byte) error {
	var alias additionalDataAlias
	if err := json.Unmarshal(b, &alias); err != nil {
		return err
	}
	*a = AdditionalData(alias)

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(b, &raw); err != nil {
		return err
	}
	if cp, ok := raw["correctness_passed"]; ok {
		var asBool bool
		if err := json.Unmarshal(cp, &asBool); err == nil {
			a.CorrectnessBool = &asBool
		} else {
			var asInt int
			if err := json.Unmarshal(cp, &asInt); err == nil {
				a.CorrectnessPassedN = &asInt
			}
		}
	}
	return nil
}

// Module is one libexpat module's pass count.
type Module struct {
	Total  int `json:"total"`
	Passed int `json:"passed"`
}

// Subscore mirrors one entry of reward.json's subscores list.
type Subscore struct {
	Subtask string   `json:"subtask"`
	Score   *float64 `json:"score"`
}

// SSIMDetail mirrors one entry of revideo-perf-opt's correctness_details list.
type SSIMDetail struct {
	SSIM float64 `json:"ssim"`
}

// Correctness mirrors the nested additional_data.correctness object used by
// dependent-type-checker and inference-system-optimization. The fields are the
// union both tasks read; cranelift reads a bare-bool "correctness_passed" key
// (decoded into AdditionalData.CorrectnessBool), not this nested object.
type Correctness struct {
	AcceptTotal    int     `json:"accept_total"`
	AcceptPassed   int     `json:"accept_passed"`
	RejectTotal    int     `json:"reject_total"`
	RejectPassed   int     `json:"reject_passed"`
	GatePassed     bool    `json:"gate_passed"`
	TokenMatchRate float64 `json:"token_match_rate"`
	ExactMatchRate float64 `json:"exact_match_rate"`
}

// rewardOrScore returns the first non-nil float pointer's value, mirroring the
// Python `a or b or 0.0` short-circuit used for reward_data.get("reward") or
// reward_data.get("score"). A value of 0.0 in Python is falsy, so a present-but-
// zero reward falls through to the next operand — this helper reproduces that.
func rewardOrScore(r *Reward) float64 {
	if r.Reward != nil && *r.Reward != 0 {
		return *r.Reward
	}
	if r.Score != nil && *r.Score != 0 {
		return *r.Score
	}
	return 0.0
}

// fp returns a pointer to f (for building *float64 returns inline).
func fp(f float64) *float64 { return &f }

// ExtractScore is the Go port of extract_score(reward_data, task). It returns
// the correctness in [0,1] and an optional speedup (nil when the task/path
// yields no speedup). A nil reward returns (0, nil), mirroring `if not
// reward_data`. The SSIM threshold for revideo-perf-opt uses the library
// default (0.99); use ExtractScoreSSIM to override.
func ExtractScore(reward *Reward, task string) (correctness float64, speedup *float64) {
	return ExtractScoreSSIM(reward, task, defaultSSIMThreshold)
}

// ExtractScoreSSIM is ExtractScore with an explicit revideo-perf-opt SSIM
// threshold (the reference CLI passes 0.95; the library default is 0.99).
func ExtractScoreSSIM(reward *Reward, task string, ssimThreshold float64) (float64, *float64) {
	if reward == nil {
		return 0.0, nil
	}
	rawReward := rewardOrScore(reward)
	ad := reward.AdditionalData
	if ad == nil {
		ad = &AdditionalData{}
	}
	ss := reward.Subscores

	switch task {
	case "git-to-zig", "dart-style-haskell", "lua-native-compiler",
		"libexpat-to-x86asm", "postgres-sqlite-wire-adapter":
		if task == "lua-native-compiler" {
			for _, s := range ss {
				if s.Subtask == "test_pass_rate" && s.Score != nil && *s.Score > 0 {
					return *s.Score, nil
				}
			}
			if ad.TestsPassed != nil && *ad.TestsPassed > 0 {
				total := 1
				if ad.TestsTotal != nil && *ad.TestsTotal > 1 {
					total = *ad.TestsTotal
				}
				return float64(*ad.TestsPassed) / float64(total), nil
			}
		}
		if task == "libexpat-to-x86asm" {
			if len(ad.Modules) > 0 {
				totalW := 0
				for _, w := range libexpatModuleWeights {
					totalW += w
				}
				weighted := 0.0
				for _, m := range libexpatModuleOrder {
					w := libexpatModuleWeights[m]
					if w == 0 {
						continue
					}
					st := ad.Modules[m]
					if st.Total > 0 {
						weighted += (float64(st.Passed) / float64(st.Total)) * float64(w)
					}
				}
				correctness := 0.0
				if totalW != 0 {
					correctness = weighted / float64(totalW)
				}
				perf := 0.0
				if len(ss) > 1 && ss[1].Score != nil {
					perf = *ss[1].Score
				}
				return correctness, fp(perf)
			}
			return 0.0, nil
		}
		return rawReward, nil

	case "ffmpeg-swscale-rewrite":
		passed := 0
		if ad.CorrectnessPassedN != nil {
			passed = *ad.CorrectnessPassedN
		}
		total := 30
		if ad.CorrectnessTotal != nil {
			total = *ad.CorrectnessTotal
		}
		if total < 1 {
			total = 1
		}
		correctness := float64(passed) / float64(total)
		var speedup *float64
		if correctness >= 1.0 {
			switch {
			case ad.GeometricMean != nil:
				speedup = fp(*ad.GeometricMean)
			case ad.GeoMeanSpeedup != nil:
				speedup = fp(*ad.GeoMeanSpeedup)
			default:
				speedup = fp(rawReward)
			}
		}
		return correctness, speedup

	case "optimizer-design":
		return rawReward, nil

	case "notebook-compression":
		if reward.Reward != nil && *reward.Reward > 0 {
			return 1.0, fp(1.0 - *reward.Reward)
		}
		return 0.0, nil

	case "revideo-perf-opt":
		return extractRevideo(reward, rawReward, ssimThreshold)

	case "cranelift-codegen-opt":
		return extractCranelift(reward, ad)

	case "dependent-type-checker":
		return extractDependentTypeChecker(reward, ad, rawReward)

	case "modular-stack-wan21":
		return extractModularStack(reward, ss, rawReward)

	case "inference-system-optimization":
		if ad.Correctness != nil {
			if ad.Correctness.TokenMatchRate >= 0.95 {
				if rawReward > 0 {
					return 1.0, fp(rawReward)
				}
				return 1.0, nil
			}
			return ad.Correctness.ExactMatchRate, nil
		}
		if rawReward > 0 {
			return 1.0, fp(rawReward)
		}
		return 0.0, nil

	case "pyright-type-checking-optimization":
		if rawReward > 0 {
			return 1.0, fp(rawReward)
		}
		if reward.ParityPassRate != nil && *reward.ParityPassRate > 0 {
			return *reward.ParityPassRate, nil
		}
		return 0.0, nil

	case "granite-mamba2-inference-optimization":
		if rawReward > 0 {
			return 1.0, fp(rawReward)
		}
		return 0.0, nil
	}

	// ml_research default (pcqm4mv2-autoresearch, frogsgame-rl).
	return rawReward, nil
}

var correctnessFailedRe = regexp.MustCompile(`(\d+)/(\d+)`)

// extractRevideo ports the revideo-perf-opt branch of extract_score.
func extractRevideo(reward *Reward, rawReward, ssimThreshold float64) (float64, *float64) {
	correctnessOK := reward.CorrectnessOK
	var geoSpeedup float64
	if reward.GeometricMean != nil {
		geoSpeedup = *reward.GeometricMean
	}

	if ssimThreshold < 0.99 {
		cd := reward.CorrectnessDetails
		if len(cd) > 0 {
			nPass := 0
			for _, s := range cd {
				if s.SSIM >= ssimThreshold {
					nPass++
				}
			}
			nTotal := len(cd)
			if nTotal > 0 && nPass == nTotal {
				if geoSpeedup != 0 {
					return 1.0, fp(geoSpeedup)
				}
				return 1.0, nil
			}
			if nTotal > 0 {
				return float64(nPass) / float64(nTotal), nil
			}
		}
		if correctnessOK && rawReward > 0 {
			return 1.0, fp(rawReward)
		}
		return 0.0, nil
	}

	if correctnessOK && rawReward > 0 {
		return 1.0, fp(rawReward)
	}
	hf := reward.HardFailReasons
	if len(hf) > 0 && geoSpeedup != 0 {
		for _, reason := range hf {
			if strings.Contains(reason, "correctness_failed:") {
				parts := strings.SplitN(reason, ":", 2)
				failing := strings.Split(parts[1], ",")
				nFail := 0
				for _, f := range failing {
					if strings.TrimSpace(f) != "" {
						nFail++
					}
				}
				cd := reward.CorrectnessDetails
				nTotal := 8
				if len(cd) > 0 {
					nTotal = len(cd)
				}
				nPass := nTotal - nFail
				if nTotal > 0 {
					return float64(nPass) / float64(nTotal), nil
				}
			}
		}
	}
	return 0.0, nil
}

// extractCranelift ports the cranelift-codegen-opt branch. additional_data
// carries a bare-bool "correctness_passed" (not the nested correctness object),
// so it is decoded from the raw map view.
func extractCranelift(reward *Reward, ad *AdditionalData) (float64, *float64) {
	if ad.CorrectnessBool != nil && *ad.CorrectnessBool {
		whm := 1.0
		if ad.WeightedHarmonic != nil {
			whm = *ad.WeightedHarmonic
		}
		return 1.0, fp(whm)
	}
	return 0.0, nil
}

// extractDependentTypeChecker ports the dependent-type-checker branch.
func extractDependentTypeChecker(reward *Reward, ad *AdditionalData, rawReward float64) (float64, *float64) {
	corr := ad.Correctness
	if corr != nil {
		total := corr.AcceptTotal + corr.RejectTotal
		passed := corr.AcceptPassed + corr.RejectPassed
		correctness := 0.0
		if total > 0 {
			correctness = float64(passed) / float64(total)
		}
		if corr.GatePassed && rawReward > 0 {
			return 1.0, fp(rawReward)
		}
		return correctness, nil
	}
	if rawReward > 0 {
		return 1.0, fp(rawReward)
	}
	return 0.0, nil
}

// extractModularStack ports the modular-stack-wan21 branch.
func extractModularStack(reward *Reward, ss []Subscore, rawReward float64) (float64, *float64) {
	for _, s := range ss {
		if strings.Contains(strings.ToLower(s.Subtask), "correctness") {
			c := 0.0
			if s.Score != nil {
				c = *s.Score
			}
			if c > 0 {
				if c >= 1.0 && rawReward > 0 {
					return c, fp(rawReward)
				}
				return c, nil
			}
		}
	}
	if m := correctnessFailedRe.FindStringSubmatch(reward.Reason); m != nil {
		num, _ := strconv.Atoi(m[1])
		den, _ := strconv.Atoi(m[2])
		if den != 0 {
			return float64(num) / float64(den), nil
		}
	}
	if rawReward > 0 {
		return 1.0, fp(rawReward)
	}
	return 0.0, nil
}

// GatedScore is the Go port of compute_gated_score(correctness, speedup, task).
// It applies the category gate to produce the leaderboard number. An unknown
// task name returns 0 (the Python raises KeyError; the Go port returns 0 so a
// caller need not handle a panic — use CategoryOf to validate the task first).
func GatedScore(correctness float64, speedup *float64, task string) float64 {
	if task == "libexpat-to-x86asm" {
		if correctness >= 1.0 && speedup != nil {
			return 0.5 + *speedup*0.5
		}
		return correctness * 0.5
	}
	cat, ok := CategoryOf(task)
	if !ok {
		return 0.0
	}
	switch cat {
	case CategoryImplementation:
		return correctness
	case CategoryPerformance:
		if task == "notebook-compression" {
			if correctness >= 1.0 && speedup != nil {
				return *speedup
			}
			return 0.0
		}
		if correctness >= 1.0 && speedup != nil {
			return 0.5 + *speedup*0.5
		}
		return correctness * 0.5
	case CategoryMLResearch:
		if task == "frogsgame-rl" {
			correctness = correctness / 500.0 // raw reward is a board count 0-500
		}
		return correctness
	}
	return 0.0
}

// GatedScoreAntiCheat applies the anti-cheat zeroing on top of GatedScore: a
// trial flagged in scoring/anticheat.json scores 0 regardless of its raw gated
// score. The reference script's compute_gated_score does not carry this flag —
// the upstream pipeline applies anti-cheat separately — so this is the runtime's
// composition point. See FRONTIERSWE-SCORING-PARITY.md.
func GatedScoreAntiCheat(correctness float64, speedup *float64, task string, antiCheatFlagged bool) float64 {
	if antiCheatFlagged {
		return 0.0
	}
	return GatedScore(correctness, speedup, task)
}

// Score is the end-to-end convenience: extract then gate, with anti-cheat
// zeroing. It is what a fak FrontierSWE run calls to self-score a trial.
func Score(reward *Reward, task string, antiCheatFlagged bool) float64 {
	correctness, speedup := ExtractScore(reward, task)
	return GatedScoreAntiCheat(correctness, speedup, task, antiCheatFlagged)
}
