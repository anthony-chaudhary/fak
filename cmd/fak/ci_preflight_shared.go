package main

import "github.com/anthony-chaudhary/fak/internal/committedtree"

// ciPreflightFailure is shared by repository validation commands that retain
// their own migration lifecycle after ci-preflight moves to fak-dev.
type ciPreflightFailure struct {
	Step   string   `json:"step"`
	Detail string   `json:"detail,omitempty"`
	Files  []string `json:"files,omitempty"`
}

func gitRevParse(repo, ref string) (string, error) { return committedtree.Resolve(repo, ref) }
func extractCommittedTip(repo, object string) (string, error) {
	return committedtree.Extract(repo, object)
}
