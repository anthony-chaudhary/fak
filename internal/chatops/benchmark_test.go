package chatops

import (
	"fmt"
	"testing"
	"time"
)

var (
	benchSinkResult   Result
	benchSinkProposal Proposal
	benchSinkRow      AuditRow
	benchSinkVerdict  Verdict
	benchSinkString   string
	benchSinkList     []Proposal
	benchSinkBool     bool
	benchSinkSpecs    []VerbSpec
	benchSinkStrings  []string
)

func benchGatedProposal() Proposal {
	res := Parse(admin("<@UBOT> dispatch #2265"), baseCfg())
	p, _, _ := Propose(res, t0, DefaultTTL)
	return p
}

func BenchmarkParse_Status(b *testing.B) {
	cfg := baseCfg()
	msg := admin("<@UBOT> status")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkResult = Parse(msg, cfg)
	}
}

func BenchmarkParse_Dispatch(b *testing.B) {
	cfg := baseCfg()
	msg := admin("<@UBOT> dispatch #2265")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkResult = Parse(msg, cfg)
	}
}

func BenchmarkParse_Approve(b *testing.B) {
	cfg := baseCfg()
	msg := admin("<@UBOT> approve a1b2c3d4")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkResult = Parse(msg, cfg)
	}
}

func BenchmarkParse_RefusalGates(b *testing.B) {
	cfg := baseCfg()
	cases := []struct {
		name string
		msg  Message
	}{
		{"BotLoop", Message{User: "UADMIN", BotID: "B1", Channel: "CTRL", Text: "<@UBOT> status"}},
		{"WrongChannel", Message{User: "UADMIN", Channel: "OTHER", Text: "<@UBOT> status"}},
		{"NotAddressed", Message{User: "UADMIN", Channel: "CTRL", Text: "status"}},
		{"NotAdmin", Message{User: "UINTRUDER", Channel: "CTRL", Text: "<@UBOT> dispatch #1"}},
	}
	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchSinkResult = Parse(tc.msg, cfg)
			}
		})
	}
}

func BenchmarkParse_AdminAllowlistScaling(b *testing.B) {
	for _, size := range []int{1, 10, 50, 200} {
		admins := make([]string, size)
		for i := 0; i < size; i++ {
			admins[i] = fmt.Sprintf("UADMIN%04d", i)
		}
		// Put matching admin at the end of the allowlist to measure worst-case linear lookup
		cfg := Config{
			BotUserID:      "UBOT",
			ControlChannel: "CTRL",
			Admins:         admins,
		}
		msg := Message{
			User:    admins[size-1],
			Channel: "CTRL",
			TS:      "1712000000.000100",
			Text:    "<@UBOT> status",
		}
		b.Run(fmt.Sprintf("%d_admins", size), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchSinkResult = Parse(msg, cfg)
			}
		})
	}
}

func BenchmarkPropose(b *testing.B) {
	cfg := baseCfg()
	res := Parse(admin("<@UBOT> dispatch #2265"), cfg)
	now := time.Date(2026, 8, 7, 12, 0, 0, 0, time.UTC)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p, row, ok := Propose(res, now, DefaultTTL)
		benchSinkProposal = p
		benchSinkRow = row
		benchSinkBool = ok
	}
}

func BenchmarkCard(b *testing.B) {
	p := Proposal{
		Nonce:     "a1b2c3d4",
		Verb:      VerbDispatch,
		Operand:   "#2265",
		Risk:      RiskOutwardFacing,
		Proposer:  "UADMIN",
		Channel:   "CTRL",
		ThreadTS:  "1754568000.000100",
		At:        t0,
		ExpiresAt: t0.Add(DefaultTTL),
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkString = Card(p)
	}
}

func BenchmarkAdjudicate_Approve_SingleOperator(b *testing.B) {
	cfg := baseCfg()
	p := benchGatedProposal()
	reply := Parse(replyMsg("UADMIN", "CTRL", "<@UBOT> approve "+p.Nonce), cfg)
	at := t0.Add(2 * time.Minute)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkVerdict = Adjudicate(reply, p, cfg, at)
	}
}

func BenchmarkAdjudicate_Approve_MultiOperator(b *testing.B) {
	twoAdmins := Config{BotUserID: "UBOT", ControlChannel: "CTRL", Admins: []string{"UADMIN", "USECOND"}}
	p := benchGatedProposal()
	reply := Parse(replyMsg("USECOND", "CTRL", "<@UBOT> approve "+p.Nonce), twoAdmins)
	at := t0.Add(2 * time.Minute)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkVerdict = Adjudicate(reply, p, twoAdmins, at)
	}
}

func BenchmarkAdjudicate_Deny(b *testing.B) {
	cfg := baseCfg()
	p := benchGatedProposal()
	reply := Parse(replyMsg("UADMIN", "CTRL", "<@UBOT> deny "+p.Nonce), cfg)
	at := t0.Add(2 * time.Minute)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkVerdict = Adjudicate(reply, p, cfg, at)
	}
}

func BenchmarkAdjudicate_RefusedReplayed(b *testing.B) {
	cfg := baseCfg()
	p := benchGatedProposal()
	p.Resolved = true
	p.Verdict = VerdictApproved
	reply := Parse(replyMsg("UADMIN", "CTRL", "<@UBOT> approve "+p.Nonce), cfg)
	at := t0.Add(2 * time.Minute)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkVerdict = Adjudicate(reply, p, cfg, at)
	}
}

func BenchmarkExecuteRow(b *testing.B) {
	cfg := baseCfg()
	p := benchGatedProposal()
	reply := Parse(replyMsg("UADMIN", "CTRL", "<@UBOT> approve "+p.Nonce), cfg)
	at := t0.Add(2 * time.Minute)
	v := Adjudicate(reply, p, cfg, at)
	execAt := at.Add(2 * time.Second)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		row, ok := v.ExecuteRow("run-4242", execAt)
		benchSinkRow = row
		benchSinkBool = ok
	}
}

func makeSyntheticJournal(numProposals int) []AuditRow {
	journal := make([]AuditRow, 0, numProposals*2)
	now := t0
	for i := 0; i < numProposals; i++ {
		nonce := fmt.Sprintf("%08x", i+1)
		proposeRow := AuditRow{
			Event:    EventPropose,
			Nonce:    nonce,
			Verb:     VerbDispatch,
			Operand:  fmt.Sprintf("#%d", 2000+i),
			Risk:     RiskOutwardFacing,
			Proposer: "UADMIN",
			Channel:  "CTRL",
			ThreadTS: fmt.Sprintf("1754568%03d.000100", i%1000),
			At:       now,
		}
		journal = append(journal, proposeRow)

		// 80% resolved (approved/denied), 20% left pending
		if i%5 != 0 {
			resEvent := EventApprove
			verdict := VerdictApproved
			if i%5 == 1 {
				resEvent = EventDeny
				verdict = VerdictDenied
			}
			resolveRow := AuditRow{
				Event:    resEvent,
				Nonce:    nonce,
				Verb:     VerbDispatch,
				Operand:  proposeRow.Operand,
				Risk:     RiskOutwardFacing,
				Proposer: "UADMIN",
				Approver: "UADMIN",
				Verdict:  verdict,
				Channel:  "CTRL",
				ThreadTS: proposeRow.ThreadTS,
				At:       now.Add(time.Minute),
			}
			journal = append(journal, resolveRow)
		}
		now = now.Add(10 * time.Second)
	}
	return journal
}

func BenchmarkPending_Replay(b *testing.B) {
	evalTime := t0.Add(30 * time.Minute)
	for _, numProposals := range []int{10, 50, 200, 1000} {
		journal := makeSyntheticJournal(numProposals)
		b.Run(fmt.Sprintf("%d_rows", len(journal)), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchSinkList = Pending(journal, evalTime)
			}
		})
	}
}

func BenchmarkTwoTurnLifecycle(b *testing.B) {
	cfg := baseCfg()
	now := t0
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Turn 1: Parse inbound message
		msg := Message{
			User:    "UADMIN",
			Channel: "CTRL",
			TS:      "1754568000.000100",
			Text:    "<@UBOT> dispatch #2265",
		}
		parsed := Parse(msg, cfg)

		// Mint proposal card
		prop, propRow, _ := Propose(parsed, now, DefaultTTL)
		card := Card(prop)
		_ = card

		// Turn 2: Operator replies with approval
		reply := Message{
			User:    "UADMIN",
			Channel: "CTRL",
			TS:      "1754568060.000200",
			Text:    "<@UBOT> approve " + prop.Nonce,
		}
		replyParsed := Parse(reply, cfg)

		// Adjudicate approval
		verdict := Adjudicate(replyParsed, prop, cfg, now.Add(time.Minute))

		// Hand off to execution
		execRow, _ := verdict.ExecuteRow("run-101", now.Add(2*time.Minute))

		benchSinkProposal = prop
		benchSinkRow = execRow
		_ = propRow
	}
}

func BenchmarkGrammar(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkSpecs = Grammar()
	}
}

func BenchmarkReasons(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkStrings = Reasons()
	}
}
