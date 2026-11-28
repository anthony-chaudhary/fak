// SPDX-License-Identifier: Apache-2.0
// Ragged verify scatter adapted from SGLang (sgl-project/sglang, Apache-2.0),
// python/sglang/srt/speculative/ragged_verify.py and
// python/sglang/srt/layers/engram.py, revision

package model

import (
	"errors"
	"fmt"
	"sort"
)

// ErrRaggedVerifyScatter marks a refused ragged-verify pack or scatter. A
// refusal returns no partial result.
var ErrRaggedVerifyScatter = errors.New("model: ragged verify scatter refused")

// RaggedVerifyPack is the packed-token-to-request mapping of one compact
// verify batch. Draft tokens from every request are packed back to back in
// request order with no per-request padding, so a packed index says nothing
// about its owner on its own: the mapping built here at pack time is the only
// way results and cache commits find their request again.
//
// The zero value is an empty pack and refuses every scatter.
type RaggedVerifyPack struct {
	// indptr[r] is the packed index of request r's first draft token and
	// indptr[r+1] one past its last; a request with no drafts has an empty span.
	indptr []int
}

// RaggedVerifyOwner locates one packed token: the request that owns it and the
// token's position inside that request's draft.
type RaggedVerifyOwner struct {
	Request  int
	Position int
}

// RaggedVerifyCommit is one request's share of a verify step: how many of its
// leading draft tokens were accepted, and the packed range holding them, which
// is the range whose verify-pass cache entries the request keeps.
type RaggedVerifyCommit struct {
	Request     int
	Accepted    int
	PackedStart int
	PackedEnd   int
}

// NewRaggedVerifyPack builds the mapping for a batch whose request r
// contributes draftLens[r] draft tokens. A request may contribute none.
func NewRaggedVerifyPack(draftLens []int) (*RaggedVerifyPack, error) {
	if len(draftLens) == 0 {
		return nil, fmt.Errorf("%w: no requests", ErrRaggedVerifyScatter)
	}
	indptr := make([]int, len(draftLens)+1)
	for r, n := range draftLens {
		if n < 0 {
			return nil, fmt.Errorf("%w: request %d has %d draft tokens", ErrRaggedVerifyScatter, r, n)
		}
		indptr[r+1] = indptr[r] + n
	}
	return &RaggedVerifyPack{indptr: indptr}, nil
}

// Requests is the number of requests in the batch, including empty ones.
func (p *RaggedVerifyPack) Requests() int {
	if p == nil || len(p.indptr) == 0 {
		return 0
	}
	return len(p.indptr) - 1
}

// Tokens is the number of real packed draft tokens.
func (p *RaggedVerifyPack) Tokens() int {
	if p.Requests() == 0 {
		return 0
	}
	return p.indptr[len(p.indptr)-1]
}

// Span returns request r's packed range [start, end).
func (p *RaggedVerifyPack) Span(r int) (start, end int, ok bool) {
	if r < 0 || r >= p.Requests() {
		return 0, 0, false
	}
	return p.indptr[r], p.indptr[r+1], true
}

// Owner is the inverse mapping for one packed token. ok is false for an index
// outside the real tokens, which is where a fixed-size verify graph keeps its
// padding.
func (p *RaggedVerifyPack) Owner(token int) (RaggedVerifyOwner, bool) {
	if token < 0 || token >= p.Tokens() {
		return RaggedVerifyOwner{}, false
	}
	// The owner is the first request whose span ends past the token; requests
	// with an empty span end at or before it and are skipped.
	r := sort.Search(p.Requests(), func(i int) bool { return p.indptr[i+1] > token })
	return RaggedVerifyOwner{Request: r, Position: token - p.indptr[r]}, true
}

// TokenRows returns the owning request of each of numTokens packed slots.
// Slots past the real tokens map to the ghost row Requests(), so a
// fixed-capacity consumer can index a table with one spare row instead of
// branching on padding.
func (p *RaggedVerifyPack) TokenRows(numTokens int) ([]int, error) {
	if p.Requests() == 0 {
		return nil, fmt.Errorf("%w: empty pack", ErrRaggedVerifyScatter)
	}
	if numTokens < p.Tokens() {
		return nil, fmt.Errorf("%w: %d slots cannot hold %d packed tokens", ErrRaggedVerifyScatter, numTokens, p.Tokens())
	}
	rows := make([]int, numTokens)
	for t := range rows {
		rows[t] = p.Requests()
	}
	for r := 0; r < p.Requests(); r++ {
		for t := p.indptr[r]; t < p.indptr[r+1]; t++ {
			rows[t] = r
		}
	}
	return rows, nil
}

// ScatterRaggedVerify returns each request's slice of a packed per-token
// result. packed may be longer than the real tokens; the padding is dropped.
// Every request gets an entry, empty for a request with no drafts, and the
// returned slices do not alias packed.
func ScatterRaggedVerify[T any](p *RaggedVerifyPack, packed []T) ([][]T, error) {
	if p.Requests() == 0 {
		return nil, fmt.Errorf("%w: empty pack", ErrRaggedVerifyScatter)
	}
	if len(packed) < p.Tokens() {
		return nil, fmt.Errorf("%w: %d packed results for %d packed tokens", ErrRaggedVerifyScatter, len(packed), p.Tokens())
	}
	out := make([][]T, p.Requests())
	for r := range out {
		out[r] = append([]T{}, packed[p.indptr[r]:p.indptr[r+1]]...)
	}
	return out, nil
}

// CommitAcceptances scatters a packed accept/reject vector back to its
// requests. A request accepts its leading run of accepted tokens: the first
// rejection ends the run, and a later accepted token of the same request is
// not committed, because it was verified against a prefix that did not
// survive. The run never crosses into the next request's span.
func (p *RaggedVerifyPack) CommitAcceptances(accepted []bool) ([]RaggedVerifyCommit, error) {
	if p.Requests() == 0 {
		return nil, fmt.Errorf("%w: empty pack", ErrRaggedVerifyScatter)
	}
	if len(accepted) < p.Tokens() {
		return nil, fmt.Errorf("%w: %d verdicts for %d packed tokens", ErrRaggedVerifyScatter, len(accepted), p.Tokens())
	}
	commits := make([]RaggedVerifyCommit, p.Requests())
	for r := range commits {
		start, end := p.indptr[r], p.indptr[r+1]
		n := 0
		for start+n < end && accepted[start+n] {
			n++
		}
		commits[r] = RaggedVerifyCommit{Request: r, Accepted: n, PackedStart: start, PackedEnd: start + n}
	}
	return commits, nil
}
