package issuehygiene

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

const DefaultPacketSize = 1

// PacketOptions controls the only unsafe planner mode. Issue mutation defaults to
// one issue per worker; batching requires an explicit override plus auditable
// price and review references.
type PacketOptions struct {
	PacketSize      int
	UnsafeOversized bool
	PriceRef        string
	ReviewRef       string
}

type WorkerPacket struct {
	ID      string   `json:"id"`
	Issues  []int    `json:"issues"`
	Defects []string `json:"defects"`
}

type PacketPlan struct {
	Schema         string         `json:"schema"`
	SourceSchema   string         `json:"source_schema"`
	PacketSize     int            `json:"packet_size"`
	IssueCount     int            `json:"issue_count"`
	DefectCount    int            `json:"defect_count"`
	UnsafeOverride bool           `json:"unsafe_override"`
	PriceRef       string         `json:"price_ref,omitempty"`
	ReviewRef      string         `json:"review_ref,omitempty"`
	Packets        []WorkerPacket `json:"packets"`
}

type issueDefects struct {
	number  int
	defects []string
}

// PlanPackets converts issue-hygiene hard defects into stable, disjoint worker
// packets. A repeated issue across KPI defect lists is assigned to exactly one
// packet while every distinct defect string remains covered.
func PlanPackets(payload scorecard.Payload, opts PacketOptions) (PacketPlan, error) {
	if payload.Schema != Schema {
		return PacketPlan{}, fmt.Errorf("source schema %q is not %q", payload.Schema, Schema)
	}
	size := opts.PacketSize
	if size == 0 {
		size = DefaultPacketSize
	}
	if size < 1 {
		return PacketPlan{}, fmt.Errorf("packet size must be at least 1")
	}
	priceRef := strings.TrimSpace(opts.PriceRef)
	reviewRef := strings.TrimSpace(opts.ReviewRef)
	if size > DefaultPacketSize {
		if !opts.UnsafeOversized {
			return PacketPlan{}, fmt.Errorf("packet size %d exceeds safe default %d; pass --unsafe-oversized with --price-ref and --review-ref", size, DefaultPacketSize)
		}
		if priceRef == "" || reviewRef == "" {
			return PacketPlan{}, fmt.Errorf("oversized packets require non-empty price and review references")
		}
	}

	byIssue := make(map[int]map[string]struct{})
	defectCount := 0
	for _, kpi := range payload.KPIs {
		for _, defect := range kpi.Defects {
			number, ok := defectIssueNumber(defect)
			if !ok {
				return PacketPlan{}, fmt.Errorf("defect does not start with an issue ID: %q", defect)
			}
			if byIssue[number] == nil {
				byIssue[number] = make(map[string]struct{})
			}
			if _, exists := byIssue[number][defect]; !exists {
				byIssue[number][defect] = struct{}{}
				defectCount++
			}
		}
	}

	issues := make([]issueDefects, 0, len(byIssue))
	for number, set := range byIssue {
		defects := make([]string, 0, len(set))
		for defect := range set {
			defects = append(defects, defect)
		}
		sort.Strings(defects)
		issues = append(issues, issueDefects{number: number, defects: defects})
	}
	sort.Slice(issues, func(i, j int) bool { return issues[i].number < issues[j].number })

	packets := make([]WorkerPacket, 0, (len(issues)+size-1)/size)
	for start := 0; start < len(issues); start += size {
		end := start + size
		if end > len(issues) {
			end = len(issues)
		}
		packet := WorkerPacket{ID: fmt.Sprintf("issue-hygiene-%04d", len(packets)+1)}
		for _, issue := range issues[start:end] {
			packet.Issues = append(packet.Issues, issue.number)
			packet.Defects = append(packet.Defects, issue.defects...)
		}
		packets = append(packets, packet)
	}

	return PacketPlan{
		Schema:         "fak-issue-hygiene-packets/1",
		SourceSchema:   payload.Schema,
		PacketSize:     size,
		IssueCount:     len(issues),
		DefectCount:    defectCount,
		UnsafeOverride: size > DefaultPacketSize,
		PriceRef:       priceRef,
		ReviewRef:      reviewRef,
		Packets:        packets,
	}, nil
}

func defectIssueNumber(defect string) (int, bool) {
	defect = strings.TrimSpace(defect)
	if len(defect) < 2 || defect[0] != '#' {
		return 0, false
	}
	end := 1
	for end < len(defect) && defect[end] >= '0' && defect[end] <= '9' {
		end++
	}
	if end == 1 || (end < len(defect) && defect[end] != ' ' && defect[end] != ':') {
		return 0, false
	}
	n, err := strconv.Atoi(defect[1:end])
	return n, err == nil && n > 0
}
