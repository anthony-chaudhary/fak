// Copyright 2026 The fak Authors
// Licensed under the Apache License, Version 2.0.
//
// DIRECT-PORT of langchain-ai/langmem src/langmem/short_term/summarization.py
// (_preprocess_messages tool-call pairing rule) at
// 8620b986af5b0c7593e6b7fbece2998690d22d15 (MIT License, Copyright (c) 2025 LangChain).
// Ported: when the summarized prefix ends on a tool call, its matching tool
// results are pulled into the prefix so a call/result pair never splits.
// fak adds: every call in the prefix is honored (not only the last item), the
// advance repeats to a fixed point, and the result is the cut CheckPrune accepts
// instead of a refusal.

package compactcohere

import "strings"

// PairSafeCut moves a proposed prefix cut forward until no tool result kept in
// transcript[cut:] answers a tool call removed in transcript[:cut]. An
// out-of-range cut is returned unchanged so CheckPrune still reports it as
// PruneInvalidCut. Results without a matching removed call, and calls with no
// result anywhere, never move the cut. Callers must recount the pruned span's
// tokens because a moved cut compacts more items.
func PairSafeCut(transcript []TranscriptItem, cut int) int {
	if cut <= 0 || cut > len(transcript) {
		return cut
	}
	removedCalls := make(map[string]struct{})
	scanned := 0
	for {
		for _, item := range transcript[scanned:cut] {
			if item.Kind == TranscriptToolCall && strings.TrimSpace(item.CallID) != "" {
				removedCalls[item.CallID] = struct{}{}
			}
		}
		scanned = cut
		next := cut
		for i := cut; i < len(transcript); i++ {
			item := transcript[i]
			if item.Kind != TranscriptToolResult || strings.TrimSpace(item.CallID) == "" {
				continue
			}
			if _, removed := removedCalls[item.CallID]; removed {
				next = i + 1
			}
		}
		// Items swept in by the advance may carry further calls whose results
		// sit later still, so iterate until the cut stops moving.
		if next == cut {
			return cut
		}
		cut = next
	}
}
