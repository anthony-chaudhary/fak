// Delivery adjudication: the egress capability floor for OUTBOUND PLATFORM MESSAGES
// (Telegram/Discord/Slack/WhatsApp/Signal), issue #2850 (Track D, #2834).
//
// WHY THIS LIVES HERE. egressfloor already owns "may this call REACH a network
// destination?" for the metadata-SSRF class. A platform-delivery adapter (the Hermes
// model) sends to a third party as a NORMAL tool/egress call — there is no floor
// between "the agent decided to send this" and "it left the box to a third party." This
// half of the leaf adds that floor: an outbound message is a GOVERNED CAPABILITY, so
// before it is sent it is (1) allowlist-checked against the permitted (platform,
// destination) set and (2) redacted of well-known secret shapes, and the decision is
// (3) captured as a witnessed DeliveryRecord (destination, redaction outcome,
// adjudication verdict). The record is produced BEFORE the send so the caller gates the
// actual delivery on record.Allowed and transmits only record.RedactedMessage.
//
// It stays a PURE, init-free classifier like the rest of this tier-1 leaf: no I/O, no
// clock, imports only the frozen ABI + stdlib (regexp/sort/strings). The caller (a
// gateway send path) owns writing the witness to a ledger and registering the refusal
// name via DeliveryReasonPairs — the same consumer-registers pattern the egress rung
// uses for ReasonEgressBlock, so this leaf needs no defconfig entry and no init.
//
// The allowlist (which platforms/targets are permitted) and the redaction (what content
// is stripped before send) are DISTINCT, both-required steps — never either/or: a
// denied message is still redacted so its raw secret never reaches the witness, and an
// allowed message is still redacted so a secret never leaves under an allowed
// destination.
package egressfloor

import (
	"regexp"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// ReasonDeliveryBlock is the refusal code a denied platform delivery cites. Like
// ReasonEgressBlock it is an OUT-OF-TREE reason (above abi.ReasonCoreMax = 1023) in the
// registered range: the closed core vocabulary in internal/abi is human-owned and
// additive-only, so this leaf reserves a code and a consumer registers its name via
// DeliveryReasonPairs (the sanctioned RegisterReason extension path). A deny cites this
// so the audit summary distinguishes a delivery-allowlist block from a generic
// POLICY_BLOCK or a metadata EGRESS_BLOCK.
const ReasonDeliveryBlock abi.ReasonCode = 1025

// ReasonDeliveryBlockName is the stable name registered for ReasonDeliveryBlock.
const ReasonDeliveryBlockName = "DELIVERY_BLOCK"

// DeliveryReasonPairs is the vocabulary a consumer hands to abi.RegisterReason so a
// delivery deny resolves to DELIVERY_BLOCK in audit output instead of UNCLASSIFIED. It
// mirrors the toolproc/taskgraph consumer-registers pattern; egressfloor itself stays
// init-free and pure.
var DeliveryReasonPairs = []struct {
	Code abi.ReasonCode
	Name string
}{
	{ReasonDeliveryBlock, ReasonDeliveryBlockName},
}

// Platform is a supported outbound delivery platform. The value is the lower-case
// platform token used both as the allowlist key and in the witnessed record.
type Platform string

const (
	PlatformTelegram Platform = "telegram"
	PlatformDiscord  Platform = "discord"
	PlatformSlack    Platform = "slack"
	PlatformWhatsApp Platform = "whatsapp"
	PlatformSignal   Platform = "signal"
)

// AnyDestination is the wildcard allowlist entry: a platform whose allowed-destination
// list contains it permits ANY destination on that platform (the platform itself is the
// unit of trust). Use it sparingly — the point of the floor is a NAMED destination set.
const AnyDestination = "*"

// Delivery is one outbound message an agent decided to send to an external platform.
// Destination is the target WITHIN the platform (a chat/channel/user id), not a network
// host — the network path is Hermes' concern; this floor governs the DECISION to egress.
type Delivery struct {
	Platform    Platform
	Destination string
	Message     string
}

// DeliveryPolicy is the egress capability floor for platform delivery. Allow is the
// destination allowlist: a platform is permitted only if it is a key, and a destination
// is permitted only if it appears in that platform's list (or the list contains
// AnyDestination). The posture is DENY BY DEFAULT — a nil/empty policy permits nothing,
// matching this leaf's deny-by-default egress philosophy — so a mis-scoped or absent
// policy fails closed rather than leaking a message to a third party.
type DeliveryPolicy struct {
	Allow map[Platform][]string
}

// permits reports whether (platform, destination) is on the allowlist.
func (p DeliveryPolicy) permits(platform Platform, destination string) bool {
	dests, ok := p.Allow[platform]
	if !ok {
		return false
	}
	for _, d := range dests {
		if d == AnyDestination || d == destination {
			return true
		}
	}
	return false
}

// DeliveryRecord is the WITNESSED record the floor produces for one Delivery, before it
// is sent. It carries exactly the three things #2850's witness names — destination,
// redaction outcome, and adjudication verdict — plus the redacted message body the
// caller should transmit in place of the original. It is a pure value; the caller owns
// persisting it to a witness ledger.
type DeliveryRecord struct {
	Platform    Platform
	Destination string

	// Allowed is the adjudication verdict: true only if the allowlist permitted the
	// (platform, destination) pair. The caller sends only when this is true.
	Allowed bool
	// Reason is the refusal code when Allowed is false (0 when allowed); ReasonLabel is
	// its human summary.
	Reason      abi.ReasonCode
	ReasonLabel string

	// RedactionCount is how many secret spans were replaced; Redactions names the
	// DISTINCT secret classes that fired (sorted, stable). RedactedMessage is the body
	// with every match replaced by a "[REDACTED:<class>]" placeholder — safe to both
	// send (when Allowed) and to record in the witness.
	RedactionCount  int
	Redactions      []string
	RedactedMessage string

	// adjudicated is the mint mark: an UNEXPORTED field only Adjudicate sets, and the
	// reason a record is a capability rather than a plain struct. The exported fields
	// above must stay exported (the witness path reads them), so nothing stops an outside
	// caller from writing `DeliveryRecord{Allowed: true, RedactedMessage: raw}` — but no
	// caller outside this package can set THIS field, so a record the floor never
	// produced stays structurally distinguishable from one it did. EnsureSendable
	// (adapter.go) is the reader: a false mark is ErrNotAdjudicated, which is how "the
	// floor ran" became a checkable property of the value instead of a convention the
	// call site had to remember.
	//
	// It is stamped on EVERY record Adjudicate returns — DENY as well as allow — so the
	// two failures stay distinct at the adapter boundary: a denied record is refused as
	// ErrDeliveryDenied ("the floor ran and said no") and never mistaken for
	// ErrNotAdjudicated ("the floor never ran at all"), which is exactly the pair an
	// operator debugging a silent non-delivery has to tell apart.
	adjudicated bool
}

// Witness renders the record as a single stable line for a delivery-witness ledger or
// log: it names WHERE (platform+destination), the VERDICT (allow/deny + reason on a
// deny), and the REDACTION OUTCOME (count + classes) — never the raw secret or the
// cleartext body. Deterministic for a given record.
func (r DeliveryRecord) Witness() string {
	var b strings.Builder
	b.WriteString("delivery platform=")
	b.WriteString(string(r.Platform))
	b.WriteString(" dest=")
	b.WriteString(r.Destination)
	if r.Allowed {
		b.WriteString(" verdict=allow")
	} else {
		b.WriteString(" verdict=deny reason=")
		b.WriteString(ReasonDeliveryBlockName)
	}
	b.WriteString(" redactions=")
	b.WriteString(itoa(r.RedactionCount))
	if len(r.Redactions) > 0 {
		b.WriteString("[")
		b.WriteString(strings.Join(r.Redactions, ","))
		b.WriteString("]")
	}
	return b.String()
}

// Adjudicate runs the delivery through the capability floor and returns the witnessed
// record. It ALWAYS redacts the message (so a denied delivery's secret never reaches the
// witness, and an allowed delivery's secret never leaves under an allowed destination),
// then applies the allowlist to set the verdict. It is pure: same policy + delivery ->
// same record, no I/O, no clock.
//
// It is also the ONLY producer of a sendable record: the returned value carries the
// unexported adjudicated mark, which is what lets an adapter (adapter.go) refuse a record
// that reached it without passing through the floor.
func (p DeliveryPolicy) Adjudicate(d Delivery) DeliveryRecord {
	redacted, count, classes := redactSecrets(d.Message)
	rec := DeliveryRecord{
		Platform:        d.Platform,
		Destination:     d.Destination,
		RedactionCount:  count,
		Redactions:      classes,
		RedactedMessage: redacted,
		adjudicated:     true,
	}
	if p.permits(d.Platform, d.Destination) {
		rec.Allowed = true
		return rec
	}
	rec.Allowed = false
	rec.Reason = ReasonDeliveryBlock
	rec.ReasonLabel = "destination not on the delivery allowlist (platform=" +
		string(d.Platform) + ", destination=" + d.Destination + ")"
	return rec
}

// secretPattern is one named secret shape the pre-send redactor strips. The set is a
// deliberate ALLOWLIST of well-known token grammars (bearer headers, provider keys,
// bot/webhook tokens, private-key blocks) rather than a scan-everything heuristic, so it
// redacts a credential without mangling ordinary prose. Applied in listed order on the
// progressively-redacted text; the classes are what surface in the witness.
type secretPattern struct {
	class string
	re    *regexp.Regexp
}

// secretPatterns are the pre-send redaction rules. Each match is replaced by
// "[REDACTED:<class>]". Kept intentionally conservative and shape-anchored to avoid
// false positives on legitimate message text.
var secretPatterns = []secretPattern{
	// PEM private-key block header — a whole key pasted into a message.
	{"private-key", regexp.MustCompile(`-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----`)},
	// Authorization: Bearer <token> (and bare "bearer <token>").
	{"bearer-token", regexp.MustCompile(`(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}`)},
	// Slack tokens: xoxb-/xoxp-/xoxa-/xoxr-/xoxs-…
	{"slack-token", regexp.MustCompile(`\bxox[baprs]-[A-Za-z0-9-]{10,}`)},
	// Telegram bot token: <digits>:<35 url-safe chars>.
	{"telegram-bot-token", regexp.MustCompile(`\b\d{6,}:[A-Za-z0-9_-]{35}\b`)},
	// GitHub tokens: ghp_/gho_/ghu_/ghs_/ghr_ + 36 base62.
	{"github-token", regexp.MustCompile(`\bgh[pousr]_[A-Za-z0-9]{36}\b`)},
	// AWS access key id.
	{"aws-access-key-id", regexp.MustCompile(`\bAKIA[0-9A-Z]{16}\b`)},
	// Generic provider secret key: sk-<20+ url-safe chars> (OpenAI-style and lookalikes).
	{"api-key", regexp.MustCompile(`\bsk-[A-Za-z0-9_-]{20,}`)},
}

// redactSecrets replaces every well-known secret span in msg with a
// "[REDACTED:<class>]" placeholder and reports how many spans were replaced and which
// distinct classes fired (sorted). A message with no secrets is returned unchanged with
// a zero count and no classes — the common case, allocation-light.
func redactSecrets(msg string) (redacted string, count int, classes []string) {
	if msg == "" {
		return "", 0, nil
	}
	out := msg
	fired := map[string]bool{}
	for _, p := range secretPatterns {
		n := 0
		out = p.re.ReplaceAllStringFunc(out, func(string) string {
			n++
			return "[REDACTED:" + p.class + "]"
		})
		if n > 0 {
			count += n
			fired[p.class] = true
		}
	}
	if len(fired) == 0 {
		return out, 0, nil
	}
	classes = make([]string, 0, len(fired))
	for c := range fired {
		classes = append(classes, c)
	}
	sort.Strings(classes)
	return out, count, classes
}

// itoa is a tiny non-negative int -> decimal string, kept local so the witness renderer
// pulls in no extra import (strconv) for the one place it is needed.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}
