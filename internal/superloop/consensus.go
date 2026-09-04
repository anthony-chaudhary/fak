package superloop

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"regexp"
	"sort"
	"strings"
)

// PatchProposal represents a proposed code patch from an agent or worker.
type PatchProposal struct {
	ID       string            `json:"id"`
	Diff     string            `json:"diff"`
	NormDiff string            `json:"norm_diff,omitempty"`
	Files    []string          `json:"files,omitempty"`
	Score    float64           `json:"score,omitempty"`
	Passed   bool              `json:"passed"`
	Metadata map[string]string `json:"metadata,omitempty"`
}

// PatchCluster groups semantically equivalent proposals by their normalized diff hash/key.
type PatchCluster struct {
	Key       string          `json:"key"`
	Proposals []PatchProposal `json:"proposals"`
	Votes     int             `json:"votes"`
}

var (
	reHeaderTimestamp = regexp.MustCompile(`\s+\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}.*$`)
	reRevision        = regexp.MustCompile(`\s+\(revision\s+\d+\)$`)
	reStandaloneTime  = regexp.MustCompile(`^\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}`)
)

// NormalizeDiff strips trailing whitespace, normalizes CRLF to LF, ignores index
// headers/timestamps, and trims blank lines.
func NormalizeDiff(diff string) string {
	diff = strings.ReplaceAll(diff, "\r\n", "\n")
	diff = strings.ReplaceAll(diff, "\r", "\n")

	rawLines := strings.Split(diff, "\n")
	var lines []string

	for _, rawLine := range rawLines {
		line := strings.TrimRight(rawLine, " \t")
		if line == "" {
			continue
		}
		if shouldIgnoreDiffLine(line) {
			continue
		}
		line = stripDiffHeaderTimestamp(line)
		lines = append(lines, line)
	}

	return strings.Join(lines, "\n")
}

func shouldIgnoreDiffLine(line string) bool {
	trimmed := strings.TrimSpace(line)
	if strings.HasPrefix(trimmed, "index ") ||
		strings.HasPrefix(trimmed, "diff --git ") ||
		strings.HasPrefix(trimmed, "old mode ") ||
		strings.HasPrefix(trimmed, "new mode ") ||
		strings.HasPrefix(trimmed, "similarity index ") ||
		strings.HasPrefix(trimmed, "dissimilarity index ") {
		return true
	}
	if reStandaloneTime.MatchString(trimmed) {
		return true
	}
	return false
}

func stripDiffHeaderTimestamp(line string) string {
	if strings.HasPrefix(line, "--- ") || strings.HasPrefix(line, "+++ ") || strings.HasPrefix(line, "*** ") {
		if idx := strings.IndexByte(line, '\t'); idx != -1 {
			line = line[:idx]
		} else {
			line = reHeaderTimestamp.ReplaceAllString(line, "")
			line = reRevision.ReplaceAllString(line, "")
		}
		line = strings.TrimRight(line, " \t")
	}
	return line
}

// ClusterAndVotePatches filters proposals that passed tests (Passed == true),
// normalizes each proposal diff and groups into clusters by normalized diff hash/key,
// ranks clusters by vote count descending (ties broken by highest score or proposal ID),
// and returns the top representative proposal from the majority cluster.
func ClusterAndVotePatches(proposals []PatchProposal) (*PatchProposal, error) {
	var passed []PatchProposal
	for _, p := range proposals {
		if p.Passed {
			passed = append(passed, p)
		}
	}
	if len(passed) == 0 {
		return nil, errors.New("no passing proposals found")
	}

	clusters := ClusterPatches(passed)
	if len(clusters) == 0 || len(clusters[0].Proposals) == 0 {
		return nil, errors.New("no valid patch clusters formed")
	}

	winner := clusters[0].Proposals[0]
	return &winner, nil
}

// ClusterPatches groups proposals into clusters by normalized diff hash/key.
// Proposals within each cluster are ranked by Score descending (tie-broken by ID ascending).
// Clusters are ranked by vote count descending (tie-broken by highest score descending,
// then proposal ID ascending).
func ClusterPatches(proposals []PatchProposal) []PatchCluster {
	clusterMap := make(map[string]*PatchCluster)
	var clusterKeys []string

	for _, p := range proposals {
		p.NormDiff = NormalizeDiff(p.Diff)
		h := sha256.Sum256([]byte(p.NormDiff))
		key := hex.EncodeToString(h[:])

		cluster, exists := clusterMap[key]
		if !exists {
			cluster = &PatchCluster{
				Key: key,
			}
			clusterMap[key] = cluster
			clusterKeys = append(clusterKeys, key)
		}
		cluster.Proposals = append(cluster.Proposals, p)
		cluster.Votes++
	}

	var result []PatchCluster
	for _, key := range clusterKeys {
		cluster := clusterMap[key]
		sort.SliceStable(cluster.Proposals, func(i, j int) bool {
			if cluster.Proposals[i].Score != cluster.Proposals[j].Score {
				return cluster.Proposals[i].Score > cluster.Proposals[j].Score
			}
			return cluster.Proposals[i].ID < cluster.Proposals[j].ID
		})
		result = append(result, *cluster)
	}

	sort.SliceStable(result, func(i, j int) bool {
		if result[i].Votes != result[j].Votes {
			return result[i].Votes > result[j].Votes
		}
		topI := result[i].Proposals[0]
		topJ := result[j].Proposals[0]
		if topI.Score != topJ.Score {
			return topI.Score > topJ.Score
		}
		if topI.ID != topJ.ID {
			return topI.ID < topJ.ID
		}
		return result[i].Key < result[j].Key
	})

	return result
}
