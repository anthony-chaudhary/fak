package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
)

type microTreeBudget struct {
	Used  int64 `json:"used"`
	Limit int64 `json:"limit"`
}

type microTreeNode struct {
	ID            string          `json:"id"`
	ParentID      string          `json:"parent_id,omitempty"`
	Turns         microTreeBudget `json:"turns"`
	Tokens        microTreeBudget `json:"tokens"`
	CostMicroUSD  microTreeBudget `json:"cost_microusd"`
	Capabilities  []string        `json:"capabilities"`
	ReceiptBytes  int64           `json:"receipt_bytes"`
	VerifierState string          `json:"verifier_state"`
	RootTokens    int64           `json:"root_input_tokens"`
}

type microTreeReport struct {
	Nodes []microTreeNode `json:"nodes"`
}

func cmdMicroTree(args []string) {
	fs := flag.NewFlagSet("micro tree", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	in := fs.String("in", "-", "read a receipt-only task tree from this JSON file (- for stdin)")
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, "usage: fak micro tree [--in <report.json>|-]")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return
	}
	if fs.NArg() != 0 {
		fs.Usage()
		return
	}

	var r io.Reader = os.Stdin
	if *in != "-" {
		f, err := os.Open(*in)
		must(err)
		defer f.Close()
		r = f
	}
	var report microTreeReport
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	must(dec.Decode(&report))
	rendered, err := microTreeText(report)
	must(err)
	fmt.Print(rendered)
}

func microTreeText(report microTreeReport) (string, error) {
	if len(report.Nodes) == 0 {
		return "", errors.New("micro tree: REFUSE empty task tree")
	}
	byID := make(map[string]microTreeNode, len(report.Nodes))
	children := make(map[string][]string, len(report.Nodes))
	roots := make([]string, 0, 1)
	for _, node := range report.Nodes {
		if node.ID == "" {
			return "", errors.New("micro tree: REFUSE node id is required")
		}
		if _, exists := byID[node.ID]; exists {
			return "", fmt.Errorf("micro tree: REFUSE duplicate node %q", node.ID)
		}
		if err := validateMicroTreeNode(node); err != nil {
			return "", fmt.Errorf("micro tree: REFUSE node %q: %w", node.ID, err)
		}
		byID[node.ID] = node
		if node.ParentID == "" {
			roots = append(roots, node.ID)
		} else {
			children[node.ParentID] = append(children[node.ParentID], node.ID)
		}
	}
	if len(roots) != 1 {
		return "", fmt.Errorf("micro tree: REFUSE want exactly one root, got %d", len(roots))
	}
	for _, node := range report.Nodes {
		if node.ParentID == "" {
			continue
		}
		parent, ok := byID[node.ParentID]
		if !ok {
			return "", fmt.Errorf("micro tree: REFUSE node %q has missing parent %q", node.ID, node.ParentID)
		}
		if node.Tokens.Limit > parent.Tokens.Limit || node.CostMicroUSD.Limit > parent.CostMicroUSD.Limit {
			return "", fmt.Errorf("micro tree: REFUSE node %q expands parent budget", node.ID)
		}
		if !microChildEnvelopeAllowed(node.Capabilities, parent.Capabilities) {
			return "", fmt.Errorf("micro tree: REFUSE node %q expands parent capabilities", node.ID)
		}
	}
	for parent := range children {
		sort.Strings(children[parent])
	}

	var out strings.Builder
	out.WriteString("MICRO TASK TREE (receipt-only; child transcripts excluded)\n")
	visited := make(map[string]bool, len(report.Nodes))
	var walk func(string, int) error
	walk = func(id string, depth int) error {
		if visited[id] {
			return fmt.Errorf("micro tree: REFUSE cycle at node %q", id)
		}
		visited[id] = true
		n := byID[id]
		caps := "-"
		if len(n.Capabilities) > 0 {
			caps = strings.Join(n.Capabilities, ",")
		}
		fmt.Fprintf(&out, "%s%s depth=%d turns=%d/%d tokens=%d/%d cost_microusd=%d/%d capabilities=%s receipt_bytes=%d verifier=%s %s=%d\n",
			strings.Repeat("  ", depth), n.ID, depth, n.Turns.Used, n.Turns.Limit, n.Tokens.Used, n.Tokens.Limit,
			n.CostMicroUSD.Used, n.CostMicroUSD.Limit, caps, n.ReceiptBytes, n.VerifierState, "root_"+"context_tokens", n.RootTokens)
		for _, child := range children[id] {
			if err := walk(child, depth+1); err != nil {
				return err
			}
		}
		return nil
	}
	if err := walk(roots[0], 0); err != nil {
		return "", err
	}
	if len(visited) != len(report.Nodes) {
		return "", errors.New("micro tree: REFUSE disconnected or cyclic nodes")
	}
	return out.String(), nil
}

func validateMicroTreeNode(node microTreeNode) error {
	budgets := []struct {
		name string
		b    microTreeBudget
	}{{"turns", node.Turns}, {"tokens", node.Tokens}, {"cost_microusd", node.CostMicroUSD}}
	for _, budget := range budgets {
		if budget.b.Used < 0 || budget.b.Limit < 0 || budget.b.Used > budget.b.Limit {
			return fmt.Errorf("invalid %s budget %d/%d", budget.name, budget.b.Used, budget.b.Limit)
		}
	}
	if node.ReceiptBytes < 0 || node.RootTokens < 0 || node.VerifierState == "" {
		return errors.New("receipt size, root context contribution, and verifier state are required")
	}
	return nil
}

func microChildEnvelopeAllowed(child, parent []string) bool {
	allowed := make(map[string]struct{}, len(parent))
	for _, capability := range parent {
		allowed[capability] = struct{}{}
	}
	for _, capability := range child {
		if _, ok := allowed[capability]; !ok {
			return false
		}
	}
	return true
}
