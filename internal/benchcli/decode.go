package benchcli

import "github.com/anthony-chaudhary/fak/internal/model"

// DecodeLCG advances a session through steps deterministic decode tokens and
// returns the token that follows the last step. The benchmark commands use this
// exact recurrence to keep serial-decode work byte-identical across their arms.
func DecodeLCG(session *model.Session, token, steps, vocab int) int {
	for steps > 0 {
		session.Step(token)
		token = (token*48271 + 1) % vocab
		steps--
	}
	return token
}
