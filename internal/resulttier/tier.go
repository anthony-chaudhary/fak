package resulttier

import (
	"encoding/base64"
	"errors"
	"fmt"
	"strconv"
	"strings"
)

// StandardTier represents one of the standard bounded result tiers.
type StandardTier int

const (
	Tier1           StandardTier = 1
	Tier3           StandardTier = 3
	Tier5           StandardTier = 5
	Tier10          StandardTier = 10
	DefaultTier     StandardTier = Tier5
	MaxStandardTier StandardTier = Tier10
)

// Int returns the tier value as an int.
func (t StandardTier) Int() int {
	return int(t)
}

// ErrTierWideningRefused is returned when a pagination request exceeds limit 10 without an explicit widening reason.
var ErrTierWideningRefused = errors.New("TIER_WIDENING_REFUSED: limits > 10 require an explicit widening_reason. Inspect the initial bounded view (default tier 5) and paginate with offset/cursor")

// PaginationRequest specifies the window and tier constraints for paginated queries.
type PaginationRequest struct {
	Limit          int    `json:"limit"`
	Offset         int    `json:"offset"`
	Cursor         string `json:"cursor,omitempty"`
	WideningReason string `json:"widening_reason,omitempty"`
}

// ContinuationResponse carries the pagination metadata and continuation state.
type ContinuationResponse struct {
	TotalAvailable int    `json:"total_available"`
	ReturnedCount  int    `json:"returned_count"`
	Offset         int    `json:"offset"`
	NextCursor     string `json:"next_cursor,omitempty"`
	HasMore        bool   `json:"has_more"`
	Tier           int    `json:"tier"`
	Truncated      bool   `json:"truncated"`
}

// ResolvePagination resolves limit, offset, and continuation metadata against the total item count.
//
// Rules:
//   - If req.Limit <= 0: defaults to DefaultTier (5).
//   - If req.Limit in {1, 3, 5, 10} or <= 10: allowed without reason.
//   - If req.Limit > 10: requires a non-empty WideningReason, returning ErrTierWideningRefused otherwise.
//   - Computes valid slice boundaries [start:end] and populates ContinuationResponse.
func ResolvePagination(req PaginationRequest, total int) (limit, offset int, continuation ContinuationResponse, err error) {
	if total < 0 {
		total = 0
	}

	limit = req.Limit
	if limit <= 0 {
		limit = int(DefaultTier)
	}

	if limit > int(MaxStandardTier) {
		if strings.TrimSpace(req.WideningReason) == "" {
			return 0, 0, ContinuationResponse{}, ErrTierWideningRefused
		}
	}

	offset = req.Offset
	if offset < 0 {
		offset = 0
	}
	if offset == 0 && req.Cursor != "" {
		parsed, parseErr := parseCursor(req.Cursor)
		if parseErr != nil {
			return 0, 0, ContinuationResponse{}, parseErr
		}
		offset = parsed
	}

	start := offset
	if start > total {
		start = total
	}
	end := start + limit
	if end > total {
		end = total
	}

	returnedCount := end - start
	hasMore := end < total
	var nextCursor string
	if hasMore {
		nextCursor = strconv.Itoa(end)
	}

	continuation = ContinuationResponse{
		TotalAvailable: total,
		ReturnedCount:  returnedCount,
		Offset:         offset,
		NextCursor:     nextCursor,
		HasMore:        hasMore,
		Tier:           limit,
		Truncated:      total > limit,
	}

	return limit, offset, continuation, nil
}

// Slice slices an in-memory slice of items according to req and returns the bounded subset and continuation metadata.
func Slice[T any](items []T, req PaginationRequest) ([]T, ContinuationResponse, error) {
	limit, offset, cont, err := ResolvePagination(req, len(items))
	if err != nil {
		return nil, ContinuationResponse{}, err
	}
	if len(items) == 0 {
		return items, cont, nil
	}
	start := offset
	if start > len(items) {
		start = len(items)
	}
	end := start + limit
	if end > len(items) {
		end = len(items)
	}
	return items[start:end], cont, nil
}

func parseCursor(cursor string) (int, error) {
	s := strings.TrimSpace(cursor)
	if s == "" {
		return 0, nil
	}
	if n, err := strconv.Atoi(s); err == nil {
		if n < 0 {
			return 0, fmt.Errorf("invalid cursor %q: offset must be non-negative", cursor)
		}
		return n, nil
	}
	for _, prefix := range []string{"offset:", "cursor:"} {
		if strings.HasPrefix(strings.ToLower(s), prefix) {
			rest := strings.TrimSpace(s[len(prefix):])
			if n, err := strconv.Atoi(rest); err == nil {
				if n < 0 {
					return 0, fmt.Errorf("invalid cursor %q: offset must be non-negative", cursor)
				}
				return n, nil
			}
		}
	}
	if data, err := base64.StdEncoding.DecodeString(s); err == nil {
		ds := strings.TrimSpace(string(data))
		if n, err := strconv.Atoi(ds); err == nil {
			if n < 0 {
				return 0, fmt.Errorf("invalid cursor %q: offset must be non-negative", cursor)
			}
			return n, nil
		}
	}
	return 0, fmt.Errorf("invalid cursor %q: must be integer offset", cursor)
}
