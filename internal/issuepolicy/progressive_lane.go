package issuepolicy

import (
	"bufio"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
)

// ProgressiveLaneResult describes the outcome of progressive lane inference.
type ProgressiveLaneResult struct {
	Lane           string   `json:"lane,omitempty"`
	Provisional    bool     `json:"provisional,omitempty"`
	Confidence     float64  `json:"confidence,omitempty"`
	SuggestedLanes []string `json:"suggested_lanes,omitempty"`
}

var (
	workspaceLanesOnce sync.Once
	workspaceLanes     []string

	defaultLaneKeywords = map[string][]string{
		"kvmmu": {
			"kvmmu", "kv cache", "kv-cache", "block manager", "evict",
			"evicting", "cache eviction", "paged attention", "block table",
			"page table", "cache blocks",
		},
		"radixkv": {
			"radixkv", "radix", "radix tree", "prefix cache", "prefix tree",
			"token tree", "radix cache",
		},
		"cachemeta": {
			"cachemeta", "cache metadata", "metadata store", "cache entries",
			"cache index", "metadata",
		},
		"gateway": {
			"gateway", "request multiplexer", "multiplexer", "reverse proxy",
			"proxy", "upstream", "http handler", "request routing",
		},
		"engine": {
			"engine", "inference engine", "model execution", "forward pass",
			"sampling", "tokenizer", "decode", "prefill",
		},
		"compute": {
			"compute", "cuda", "metal", "gemm", "matmul",
			"gpu kernel", "tensor", "simd", "acceleration", "accelerator",
		},
		"adjudicator": {
			"adjudicator", "adjudication", "decision", "verdict",
			"capability floor", "syscall filter", "tool filter", "policy gate", "sandbox",
		},
		"issueorchestrator": {
			"issueorchestrator", "wave", "wave planning", "dispatch queue",
			"issue queue", "batch dispatch", "orchestrator", "work stream", "subdivide",
		},
		"issuepolicy": {
			"issuepolicy", "issue contract", "issue policy", "draft review",
			"problem frame", "task brief", "scope check", "triage classification", "provisional lane",
		},
		"windowgate": {
			"windowgate", "window", "desktop popup", "gui window", "modal", "window focus",
		},
		"debtlane": {
			"debtlane", "debt clean", "maturity debt", "technical debt", "debt burndown",
		},
		"amdgpu": {
			"amdgpu", "rocm", "hip", "amd", "radeon",
		},
		"codetools": {
			"codetools", "linter", "formatter", "ast rewrite", "code transform", "formatting",
		},
		"modver": {
			"modver", "module version", "version skew", "rev counter", "monotonic rev",
		},
		"logvault": {
			"logvault", "log vault", "audit log", "log capture", "decision log", "hash chain",
		},
		"sessionsignals": {
			"sessionsignals", "session signals", "session monitor", "session health", "stall detector",
		},
		"workerenvelope": {
			"workerenvelope", "worker envelope", "worker budget", "worker boundary", "subagent envelope",
		},
		"agent": {
			"agent", "headless agent", "agent loop", "subagent", "autonomous agent",
		},
		"bench": {
			"bench", "benchmark", "throughput", "tok/s", "latency", "eval",
		},
		"policy": {
			"policy", "manifest", "permission", "capability", "rule",
		},
		"shipgate": {
			"shipgate", "ship stamp", "verify ship", "witness check", "release gate",
		},
		"provenance": {
			"provenance", "source trace", "audit trail", "author attribution",
		},
		"architest": {
			"architest", "dag check", "dependency graph", "layer check", "forbidden import",
		},
		"metrics": {
			"metrics", "telemetry", "prometheus", "counter", "gauge", "histogram",
		},
		"model": {
			"model", "weights", "safetensors", "checkpoint", "gguf",
		},
		"recall": {
			"recall", "memory store", "agent memory", "re-verify memory", "stale memory",
		},
		"blob": {
			"blob", "storage", "artifact", "blob store",
		},
		"vdso": {
			"vdso", "fast path", "idempotent cache", "syscall shortcut",
		},
		"allinone": {
			"allinone", "turnkey", "one touch", "fak up",
		},
		"steward": {
			"steward", "housekeeping", "repo hygiene", "stewardship",
		},
		"marketing": {
			"marketing", "launch", "announcement", "release notes",
		},
	}
)

func getWorkspaceLanes() []string {
	workspaceLanesOnce.Do(func() {
		workspaceLanes = loadLanesFromDOSToml()
	})
	return workspaceLanes
}

func loadLanesFromDOSToml() []string {
	dir, err := os.Getwd()
	if err != nil {
		return nil
	}
	for i := 0; i < 8; i++ {
		target := filepath.Join(dir, "dos.toml")
		if info, err := os.Stat(target); err == nil && !info.IsDir() {
			return parseDOSTomlLanes(target)
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	return nil
}

func parseDOSTomlLanes(path string) []string {
	f, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	inTrees := false
	var lanes []string
	seen := make(map[string]bool)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if strings.HasPrefix(line, "[") {
			inTrees = (line == "[lanes.trees]")
			continue
		}
		if inTrees {
			if idx := strings.Index(line, "="); idx > 0 {
				lane := strings.TrimSpace(line[:idx])
				if lane != "" && !seen[lane] {
					seen[lane] = true
					lanes = append(lanes, lane)
				}
			}
		}
	}
	return lanes
}

func extractWords(s string) []string {
	var words []string
	var current strings.Builder
	for _, r := range strings.ToLower(s) {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			current.WriteRune(r)
		} else {
			if current.Len() > 0 {
				words = append(words, current.String())
				current.Reset()
			}
		}
	}
	if current.Len() > 0 {
		words = append(words, current.String())
	}
	return words
}

func stemWord(w string) string {
	for _, suffix := range []string{"ing", "tion", "tions", "ers", "er", "es", "ed", "s"} {
		if strings.HasSuffix(w, suffix) && len(w)-len(suffix) >= 3 {
			return w[:len(w)-len(suffix)]
		}
	}
	return w
}

type laneScore struct {
	lane  string
	score float64
}

// InferLaneProgressive indexes known repository lanes and scores candidate issue
// titles, bodies, and path hints, returning a confidence-tiered progressive lane result.
func InferLaneProgressive(title, body string, paths []string, declaredLanes ...string) ProgressiveLaneResult {
	if exact := inferLane(title, paths); exact != "" {
		return ProgressiveLaneResult{
			Lane:        exact,
			Provisional: false,
			Confidence:  1.0,
		}
	}

	laneSet := make(map[string]bool)
	var candidates []string

	addLane := func(l string) {
		l = strings.TrimSpace(strings.ToLower(l))
		l = strings.TrimPrefix(l, "provisional:")
		if l != "" && !laneSet[l] {
			laneSet[l] = true
			candidates = append(candidates, l)
		}
	}

	for _, l := range declaredLanes {
		addLane(l)
	}
	for _, l := range getWorkspaceLanes() {
		addLane(l)
	}
	for l := range defaultLaneKeywords {
		addLane(l)
	}

	titleLower := strings.ToLower(title)
	bodyLower := strings.ToLower(body)
	titleWords := extractWords(titleLower)
	bodyWords := extractWords(bodyLower)

	scores := make([]laneScore, 0, len(candidates))

	for _, lane := range candidates {
		kws := defaultLaneKeywords[lane]
		hasLaneWord := false
		for _, kw := range kws {
			if kw == lane {
				hasLaneWord = true
				break
			}
		}
		if !hasLaneWord {
			kws = append([]string{lane}, kws...)
		}

		matchedTitleWords := make(map[int]bool)
		matchedBodyWords := make(map[int]bool)
		titleMatches := 0
		bodyMatches := 0
		titleScore := 0.0
		bodyScore := 0.0

		for _, kw := range kws {
			kwLower := strings.ToLower(kw)
			if strings.Contains(kwLower, " ") || strings.Contains(kwLower, "-") {
				if strings.Contains(titleLower, kwLower) {
					titleScore += 0.30
					titleMatches++
				} else if strings.Contains(bodyLower, kwLower) {
					bodyScore += 0.15
					bodyMatches++
				}
			} else {
				kwStem := stemWord(kwLower)
				matchedInTitle := false
				for i, w := range titleWords {
					if matchedTitleWords[i] {
						continue
					}
					if w == kwLower {
						if kwLower == lane {
							titleScore += 0.35
						} else {
							titleScore += 0.15
						}
						titleMatches++
						matchedTitleWords[i] = true
						matchedInTitle = true
						break
					} else if stemWord(w) == kwStem {
						titleScore += 0.12
						titleMatches++
						matchedTitleWords[i] = true
						matchedInTitle = true
						break
					}
				}

				if !matchedInTitle {
					for i, w := range bodyWords {
						if matchedBodyWords[i] {
							continue
						}
						if w == kwLower {
							if kwLower == lane {
								bodyScore += 0.18
							} else {
								bodyScore += 0.08
							}
							bodyMatches++
							matchedBodyWords[i] = true
							break
						} else if stemWord(w) == kwStem {
							bodyScore += 0.06
							bodyMatches++
							matchedBodyWords[i] = true
							break
						}
					}
				}
			}
		}

		// Token proximity / multi-match bonus: only when distinct keywords match
		if titleMatches >= 2 {
			titleScore += 0.10
		}
		if titleMatches >= 3 {
			titleScore += 0.05
		}
		if bodyMatches >= 2 {
			bodyScore += 0.05
		}

		rawScore := titleScore + bodyScore
		conf := math.Min(0.99, rawScore)
		conf = math.Round(conf*100) / 100

		scores = append(scores, laneScore{
			lane:  lane,
			score: conf,
		})
	}

	sort.SliceStable(scores, func(i, j int) bool {
		if scores[i].score != scores[j].score {
			return scores[i].score > scores[j].score
		}
		return scores[i].lane < scores[j].lane
	})

	var topSuggestions []string
	for _, s := range scores {
		if len(topSuggestions) >= 3 {
			break
		}
		topSuggestions = append(topSuggestions, s.lane)
	}
	defaultSuggestions := []string{"codetools", "steward", "marketing"}
	for _, d := range defaultSuggestions {
		if len(topSuggestions) >= 3 {
			break
		}
		found := false
		for _, s := range topSuggestions {
			if s == d {
				found = true
				break
			}
		}
		if !found {
			topSuggestions = append(topSuggestions, d)
		}
	}

	if len(scores) == 0 {
		return ProgressiveLaneResult{
			Lane:           "",
			Provisional:    false,
			Confidence:     0.0,
			SuggestedLanes: topSuggestions,
		}
	}

	best := scores[0]

	if best.score >= 0.85 {
		return ProgressiveLaneResult{
			Lane:           best.lane,
			Provisional:    false,
			Confidence:     best.score,
			SuggestedLanes: topSuggestions,
		}
	}

	if best.score >= 0.40 {
		return ProgressiveLaneResult{
			Lane:           "provisional:" + best.lane,
			Provisional:    true,
			Confidence:     best.score,
			SuggestedLanes: topSuggestions,
		}
	}

	return ProgressiveLaneResult{
		Lane:           "",
		Provisional:    false,
		Confidence:     best.score,
		SuggestedLanes: topSuggestions,
	}
}
