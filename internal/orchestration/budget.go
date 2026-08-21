package orchestration

import (
	"fmt"
	"strings"
)

const (
	BudgetLedgerSchema = "fak-orchestration-budget-ledger/1"
	RootBudgetNodeID   = "root"
)

type BudgetEventKind string

const (
	BudgetReserve BudgetEventKind = "reserve"
	BudgetConsume BudgetEventKind = "consume"
	BudgetRelease BudgetEventKind = "release"
	BudgetCancel  BudgetEventKind = "cancel"
	BudgetClose   BudgetEventKind = "close"
)

type BudgetNodeState string

const (
	BudgetNodeActive    BudgetNodeState = "active"
	BudgetNodeReleased  BudgetNodeState = "released"
	BudgetNodeCancelled BudgetNodeState = "cancelled"
	BudgetNodeClosed    BudgetNodeState = "closed"
)

type BudgetRefusalReason string

const (
	BudgetInvalidRoot             BudgetRefusalReason = "BUDGET_INVALID_ROOT"
	BudgetInvalidEvent            BudgetRefusalReason = "BUDGET_INVALID_EVENT"
	BudgetNegativeAmount          BudgetRefusalReason = "BUDGET_NEGATIVE_AMOUNT"
	BudgetDuplicateNode           BudgetRefusalReason = "BUDGET_DUPLICATE_NODE"
	BudgetParentUnknown           BudgetRefusalReason = "BUDGET_PARENT_UNKNOWN"
	BudgetParentTerminal          BudgetRefusalReason = "BUDGET_PARENT_TERMINAL"
	BudgetNodeUnknown             BudgetRefusalReason = "BUDGET_NODE_UNKNOWN"
	BudgetNodeTerminal            BudgetRefusalReason = "BUDGET_NODE_TERMINAL"
	BudgetRootExhausted           BudgetRefusalReason = "BUDGET_ROOT_EXHAUSTED"
	BudgetParentExhausted         BudgetRefusalReason = "BUDGET_PARENT_EXHAUSTED"
	BudgetConsumeExceedsRemaining BudgetRefusalReason = "BUDGET_CONSUME_EXCEEDS_REMAINING"
	BudgetActiveChildren          BudgetRefusalReason = "BUDGET_ACTIVE_CHILDREN"
)

type BudgetRefusal struct {
	Reason     BudgetRefusalReason `json:"reason"`
	EventIndex int                 `json:"event_index"`
	NodeID     string              `json:"node_id,omitempty"`
	Detail     string              `json:"detail"`
}

func (r *BudgetRefusal) Error() string {
	return fmt.Sprintf("%s: event %d node %q: %s", r.Reason, r.EventIndex, r.NodeID, r.Detail)
}

type BudgetEvent struct {
	Kind     BudgetEventKind `json:"kind"`
	NodeID   string          `json:"node_id"`
	ParentID string          `json:"parent_id,omitempty"`
	Workers  int             `json:"workers,omitempty"`
	Tokens   int64           `json:"tokens,omitempty"`
}

type BudgetTotals struct {
	Limit     Budget `json:"limit"`
	Remaining Budget `json:"remaining"`
	Reserved  Budget `json:"reserved"`
	Consumed  Budget `json:"consumed"`
}

type BudgetNode struct {
	ID         string          `json:"id"`
	ParentID   string          `json:"parent_id"`
	Allocation Budget          `json:"allocation"`
	Remaining  Budget          `json:"remaining"`
	Reserved   Budget          `json:"reserved"`
	Consumed   Budget          `json:"consumed"`
	Refunded   Budget          `json:"refunded"`
	State      BudgetNodeState `json:"state"`
}

type BudgetLedger struct {
	Schema string                `json:"schema"`
	Root   BudgetTotals          `json:"root"`
	Nodes  map[string]BudgetNode `json:"nodes"`
}

// FoldBudgetEvents deterministically reconstructs hierarchical allocations. A
// nested reservation moves capacity out of its direct parent's remaining
// balance; only top-level reservations move the root balance. Consumption is
// rolled through every ancestor so grandchildren remain visible at the root.
func FoldBudgetEvents(limit Budget, events []BudgetEvent) (BudgetLedger, error) {
	root := BudgetTotals{Limit: limit, Remaining: limit}
	if limit.MaxWorkers <= 0 || limit.MaxTokens <= 0 {
		return BudgetLedger{}, budgetRefusal(BudgetInvalidRoot, -1, RootBudgetNodeID, "worker and token limits must be positive")
	}

	nodes := make(map[string]*BudgetNode)
	children := make(map[string]map[string]struct{})
	for i, event := range events {
		event.NodeID = strings.TrimSpace(event.NodeID)
		event.ParentID = strings.TrimSpace(event.ParentID)
		amount := Budget{MaxWorkers: event.Workers, MaxTokens: event.Tokens}
		if amount.MaxWorkers < 0 || amount.MaxTokens < 0 {
			return BudgetLedger{}, budgetRefusal(BudgetNegativeAmount, i, event.NodeID, "workers and tokens must be non-negative")
		}

		var err error
		switch event.Kind {
		case BudgetReserve:
			err = foldBudgetReserve(&root, nodes, children, i, event, amount)
		case BudgetConsume:
			err = foldBudgetConsume(&root, nodes, i, event, amount)
		case BudgetRelease, BudgetCancel, BudgetClose:
			err = foldBudgetTerminal(&root, nodes, children, i, event, amount)
		default:
			err = budgetRefusal(BudgetInvalidEvent, i, event.NodeID, fmt.Sprintf("unknown event kind %q", event.Kind))
		}
		if err != nil {
			return BudgetLedger{}, err
		}
	}

	result := BudgetLedger{Schema: BudgetLedgerSchema, Root: root, Nodes: make(map[string]BudgetNode, len(nodes))}
	for id, node := range nodes {
		result.Nodes[id] = *node
	}
	return result, nil
}

func foldBudgetReserve(root *BudgetTotals, nodes map[string]*BudgetNode, children map[string]map[string]struct{}, index int, event BudgetEvent, amount Budget) error {
	if event.NodeID == "" || event.NodeID == RootBudgetNodeID || event.ParentID == "" || event.ParentID == event.NodeID || zeroBudget(amount) {
		return budgetRefusal(BudgetInvalidEvent, index, event.NodeID, "reserve requires distinct node and parent ids plus a non-zero amount")
	}
	if _, exists := nodes[event.NodeID]; exists {
		return budgetRefusal(BudgetDuplicateNode, index, event.NodeID, "node ids cannot be reused")
	}

	if event.ParentID == RootBudgetNodeID {
		if !budgetFits(root.Remaining, amount) {
			return budgetRefusal(BudgetRootExhausted, index, event.NodeID, "reservation exceeds the root remaining balance")
		}
		root.Remaining = subtractBudget(root.Remaining, amount)
		root.Reserved = addBudget(root.Reserved, amount)
	} else {
		parent, ok := nodes[event.ParentID]
		if !ok {
			return budgetRefusal(BudgetParentUnknown, index, event.NodeID, fmt.Sprintf("parent %q is not reserved", event.ParentID))
		}
		if parent.State != BudgetNodeActive {
			return budgetRefusal(BudgetParentTerminal, index, event.NodeID, fmt.Sprintf("parent %q is %s", event.ParentID, parent.State))
		}
		if !budgetFits(parent.Remaining, amount) {
			return budgetRefusal(BudgetParentExhausted, index, event.NodeID, fmt.Sprintf("reservation exceeds parent %q remaining balance", event.ParentID))
		}
		parent.Remaining = subtractBudget(parent.Remaining, amount)
		parent.Reserved = addBudget(parent.Reserved, amount)
	}

	nodes[event.NodeID] = &BudgetNode{
		ID:         event.NodeID,
		ParentID:   event.ParentID,
		Allocation: amount,
		Remaining:  amount,
		State:      BudgetNodeActive,
	}
	if children[event.ParentID] == nil {
		children[event.ParentID] = make(map[string]struct{})
	}
	children[event.ParentID][event.NodeID] = struct{}{}
	return nil
}

func foldBudgetConsume(root *BudgetTotals, nodes map[string]*BudgetNode, index int, event BudgetEvent, amount Budget) error {
	if event.NodeID == "" || event.ParentID != "" || zeroBudget(amount) {
		return budgetRefusal(BudgetInvalidEvent, index, event.NodeID, "consume requires a node id, no parent id, and a non-zero amount")
	}
	node, ok := nodes[event.NodeID]
	if !ok {
		return budgetRefusal(BudgetNodeUnknown, index, event.NodeID, "consume requires an existing reservation")
	}
	if node.State != BudgetNodeActive {
		return budgetRefusal(BudgetNodeTerminal, index, event.NodeID, fmt.Sprintf("node is %s", node.State))
	}
	if !budgetFits(node.Remaining, amount) {
		return budgetRefusal(BudgetConsumeExceedsRemaining, index, event.NodeID, "consumption exceeds the node remaining balance")
	}

	node.Remaining = subtractBudget(node.Remaining, amount)
	node.Consumed = addBudget(node.Consumed, amount)
	for parentID := node.ParentID; ; {
		if parentID == RootBudgetNodeID {
			root.Reserved = subtractBudget(root.Reserved, amount)
			root.Consumed = addBudget(root.Consumed, amount)
			break
		}
		parent := nodes[parentID]
		parent.Reserved = subtractBudget(parent.Reserved, amount)
		parent.Consumed = addBudget(parent.Consumed, amount)
		parentID = parent.ParentID
	}
	return nil
}

func foldBudgetTerminal(root *BudgetTotals, nodes map[string]*BudgetNode, children map[string]map[string]struct{}, index int, event BudgetEvent, amount Budget) error {
	if event.NodeID == "" || event.ParentID != "" || !zeroBudget(amount) {
		return budgetRefusal(BudgetInvalidEvent, index, event.NodeID, "terminal events require only a node id")
	}
	node, ok := nodes[event.NodeID]
	if !ok {
		return budgetRefusal(BudgetNodeUnknown, index, event.NodeID, "terminal event requires an existing reservation")
	}
	if node.State != BudgetNodeActive {
		return nil
	}
	if len(children[node.ID]) != 0 {
		return budgetRefusal(BudgetActiveChildren, index, event.NodeID, "terminal nodes must close their active children first")
	}

	workerReturn := node.Remaining.MaxWorkers + node.Consumed.MaxWorkers
	if node.ParentID == RootBudgetNodeID {
		root.Remaining = addBudget(root.Remaining, Budget{MaxWorkers: workerReturn, MaxTokens: node.Remaining.MaxTokens})
		root.Reserved = subtractBudget(root.Reserved, node.Remaining)
		root.Consumed.MaxWorkers -= node.Consumed.MaxWorkers
	} else {
		parent := nodes[node.ParentID]
		parent.Remaining = addBudget(parent.Remaining, Budget{MaxWorkers: workerReturn, MaxTokens: node.Remaining.MaxTokens})
		parent.Reserved = subtractBudget(parent.Reserved, node.Remaining)
		parent.Consumed.MaxWorkers -= node.Consumed.MaxWorkers
		for ancestorID := parent.ParentID; ; {
			if ancestorID == RootBudgetNodeID {
				root.Consumed.MaxWorkers -= node.Consumed.MaxWorkers
				root.Reserved.MaxWorkers += node.Consumed.MaxWorkers
				break
			}
			ancestor := nodes[ancestorID]
			ancestor.Consumed.MaxWorkers -= node.Consumed.MaxWorkers
			ancestor.Reserved.MaxWorkers += node.Consumed.MaxWorkers
			ancestorID = ancestor.ParentID
		}
	}

	node.Refunded = Budget{MaxWorkers: workerReturn, MaxTokens: node.Remaining.MaxTokens}
	node.Remaining = Budget{}
	node.Reserved = Budget{}
	node.Consumed.MaxWorkers = 0
	switch event.Kind {
	case BudgetRelease:
		node.State = BudgetNodeReleased
	case BudgetCancel:
		node.State = BudgetNodeCancelled
	case BudgetClose:
		node.State = BudgetNodeClosed
	}
	delete(children[node.ParentID], node.ID)
	return nil
}

func budgetRefusal(reason BudgetRefusalReason, index int, nodeID, detail string) *BudgetRefusal {
	return &BudgetRefusal{Reason: reason, EventIndex: index, NodeID: nodeID, Detail: detail}
}

func zeroBudget(b Budget) bool {
	return b.MaxWorkers == 0 && b.MaxTokens == 0
}

func budgetFits(remaining, amount Budget) bool {
	return amount.MaxWorkers <= remaining.MaxWorkers && amount.MaxTokens <= remaining.MaxTokens
}

func addBudget(a, b Budget) Budget {
	return Budget{MaxWorkers: a.MaxWorkers + b.MaxWorkers, MaxTokens: a.MaxTokens + b.MaxTokens}
}

func subtractBudget(a, b Budget) Budget {
	return Budget{MaxWorkers: a.MaxWorkers - b.MaxWorkers, MaxTokens: a.MaxTokens - b.MaxTokens}
}
