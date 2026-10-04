//go:build !(darwin && arm64 && cgo)

package model

// tryDenseQ4KMetalDecodeGraph declines on the portable build: there is no Metal graph,
// so every token takes the blockStep path.
func (s *Session) tryDenseQ4KMetalDecodeGraph(id, pos int) ([]float32, bool) { return nil, false }
