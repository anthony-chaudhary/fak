package lifecycle

import (
	"errors"
	"sort"
)

// Transaction is the durable operator-control view of a managed transition.
type Transaction struct {
	ID           string   `json:"transaction_id"`
	Stage        string   `json:"stage"`
	Members      []Member `json:"members"`
	OperatorHold bool     `json:"operator_hold,omitempty"`
	Outcome      string   `json:"outcome,omitempty"`
}

type Member struct {
	ID       string `json:"id"`
	State    string `json:"state"`
	Readback bool   `json:"independent_readback,omitempty"`
}

type Action struct {
	MemberID  string `json:"member_id"`
	Operation string `json:"operation"`
}
type Preview struct {
	TransactionID string   `json:"transaction_id"`
	Control       string   `json:"control"`
	Actions       []Action `json:"actions"`
	NextStage     string   `json:"next_stage"`
	OperatorHold  bool     `json:"operator_hold"`
	Outcome       string   `json:"outcome,omitempty"`
}

// Control plans an idempotent operator action without mutating tx.
func Control(tx Transaction, control string) (Preview, error) {
	p := Preview{TransactionID: tx.ID, Control: control, Actions: []Action{}, NextStage: tx.Stage, OperatorHold: tx.OperatorHold, Outcome: tx.Outcome}
	if tx.ID == "" {
		return p, errors.New("transaction_id required")
	}
	add := func(state, op string) {
		for _, m := range tx.Members {
			if m.State == state {
				p.Actions = append(p.Actions, Action{m.ID, op})
			}
		}
	}
	switch control {
	case "inspect":
	case "cancel":
		if tx.Outcome == "cancelled" {
			break
		}
		switch tx.Stage {
		case "prepare", "partial_pause":
			add("paused", "resume")
			p.NextStage = "cancelled"
			p.Outcome = "cancelled"
		case "host_action", "partial_restore":
			add("paused", "restore")
			add("missing", "restore")
			if len(p.Actions) > 0 {
				p.NextStage = "operator_hold"
				p.OperatorHold = true
				p.Outcome = ""
			} else {
				p.NextStage = "cancelled"
				p.Outcome = "cancelled"
			}
		case "ready":
			add("paused", "resume")
			p.NextStage = "cancelled"
			p.Outcome = "cancelled"
		case "operator_hold":
			p.OperatorHold = true
		default:
			return p, errors.New("cancel unsupported for stage " + tx.Stage)
		}
	case "rollback":
		if tx.Outcome == "rolled_back" {
			break
		}
		for _, m := range tx.Members {
			if m.State != "restored" || !m.Readback {
				p.Actions = append(p.Actions, Action{m.ID, "restore_and_readback"})
			}
		}
		if len(p.Actions) > 0 {
			p.NextStage = "operator_hold"
			p.OperatorHold = true
			p.Outcome = ""
		} else {
			p.NextStage = "rolled_back"
			p.OperatorHold = false
			p.Outcome = "rolled_back"
		}
	case "retry":
		if tx.Stage != "operator_hold" && tx.Stage != "partial_restore" {
			return p, errors.New("retry requires operator_hold or partial_restore")
		}
		add("missing", "restore")
		add("paused", "restore")
		p.OperatorHold = len(p.Actions) > 0
		if p.OperatorHold {
			p.NextStage = "operator_hold"
		} else {
			p.NextStage = "ready"
		}
	case "resume":
		add("paused", "resume")
		add("restored", "verify_ready")
		p.NextStage = "ready"
		p.OperatorHold = false
	default:
		return p, errors.New("unknown control " + control)
	}
	sort.Slice(p.Actions, func(i, j int) bool {
		if p.Actions[i].MemberID == p.Actions[j].MemberID {
			return p.Actions[i].Operation < p.Actions[j].Operation
		}
		return p.Actions[i].MemberID < p.Actions[j].MemberID
	})
	return p, nil
}

func Apply(tx Transaction, p Preview) Transaction {
	tx.Stage = p.NextStage
	tx.OperatorHold = p.OperatorHold
	tx.Outcome = p.Outcome
	return tx
}
