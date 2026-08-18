package main

import (
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/policy"
)

// cmd/fak/enroll.go — `fak enroll`: pin an org trust anchor on this box, show it, or
// revoke it (#5323, W5 of epic #5315).
//
// Enrollment is OPT-IN and stays opt-in. A box that never runs this verb behaves exactly
// as it does today: `internal/policy.OrgTrustAnchor` hands back options carrying no root
// key, and a signed central envelope is still refused. That is the property the verb must
// not quietly erode, so the status arm says INERT out loud rather than printing nothing —
// "no output" is how an un-enrolled box and a broken one would otherwise look identical.
//
// The anchor is operator-supplied here, deliberately. Discovering a root key FROM the org
// URL is a remote fetch with its own trust problem (a MitM'd enroll endpoint serving its
// own key is exactly the attack the no-silent-re-pin rule exists for), and that work is
// #5321. Until it lands, --root-key is REQUIRED and its absence is a refusal that names
// the gap — never a default, and never a key pulled off the wire unpinned.
//
// Exit codes follow the sibling verbs (`fak hygiene`, #5604): 2 = usage, 1 = refusal,
// 0 = the operation happened. A refusal is 1 rather than 0-with-a-warning because the
// re-pin refusal is a security outcome the caller has to be able to branch on.

func cmdEnroll(argv []string) { os.Exit(runEnroll(os.Stdout, os.Stderr, argv)) }

func runEnroll(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("enroll", flag.ContinueOnError)
	fs.SetOutput(stderr)
	orgURL := fs.String("org", "", "org policy URL to enroll with")
	rootKey := fs.String("root-key", "", "org root Ed25519 public key, base64 or a file holding it (required to enroll)")
	issuer := fs.String("issuer", "", "pin the signing identity (default: the --org URL host)")
	device := fs.String("device", "", "device identity to bind grants to (default: this host's name)")
	user := fs.String("user", "", "user identity to bind grants to (default: $USER / $USERNAME)")
	groups := fs.String("groups", "", "comma-separated group identities to bind grants to")
	revoke := fs.Bool("revoke", false, "clear the pinned anchor, returning this box to the un-enrolled posture")
	path := fs.String("path", "", "enrollment store path (default: "+policy.OrgEnrollmentPathEnv+", else the user config dir)")
	asJSON := fs.Bool("json", false, "emit the enrollment record as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}

	store := strings.TrimSpace(*path)
	if store == "" {
		store = policy.OrgEnrollmentPath()
	}

	switch {
	case *revoke:
		return runEnrollRevoke(stdout, stderr, *asJSON, store)
	case strings.TrimSpace(*orgURL) == "":
		// No --org and no --revoke: report the current posture. This arm must never
		// be silent, because "not enrolled" and "enrolled" are the two states an
		// operator is here to tell apart.
		return runEnrollStatus(stdout, stderr, *asJSON, store)
	}

	key, err := enrollRootKey(*rootKey)
	if err != nil {
		fmt.Fprintf(stderr, "fak enroll: %v\n", err)
		return 1
	}
	req := policy.OrgEnrollRequest{
		OrgURL:   strings.TrimSpace(*orgURL),
		Issuer:   strings.TrimSpace(*issuer),
		RootKey:  key,
		DeviceID: strings.TrimSpace(*device),
		User:     strings.TrimSpace(*user),
		Groups:   splitCSV(*groups),
		Now:      time.Now(),
	}
	if req.Issuer == "" {
		req.Issuer = orgURLHost(req.OrgURL)
	}
	if req.DeviceID == "" {
		req.DeviceID = defaultDeviceID()
	}
	if req.User == "" {
		req.User = defaultUserID()
	}

	rec, err := policy.EnrollOrg(store, req)
	if err != nil {
		fmt.Fprintf(stderr, "fak enroll: %v\n", err)
		// Name the sanctioned re-pin route. A refusal an operator cannot act on gets
		// worked around, and the workaround here would be deleting the store by hand.
		fmt.Fprintf(stderr, "  next: fak enroll --revoke   (then re-run this command)\n")
		return 1
	}
	if *asJSON {
		return enrollJSON(stdout, stderr, store, rec, true)
	}
	fmt.Fprintf(stdout, "enrolled: %s\n", rec.OrgURL)
	writeEnrollmentDetail(stdout, store, rec)
	return 0
}

func runEnrollRevoke(stdout, stderr io.Writer, asJSON bool, store string) int {
	revoked, err := policy.RevokeOrgEnrollment(store)
	if err != nil {
		fmt.Fprintf(stderr, "fak enroll: %v\n", err)
		return 1
	}
	if asJSON {
		if err := writeIndentedJSON(stdout, map[string]any{
			"store":    store,
			"revoked":  revoked,
			"enrolled": false,
		}); err != nil {
			fmt.Fprintf(stderr, "fak enroll: %v\n", err)
			return 1
		}
		return 0
	}
	if !revoked {
		// Not an error: "make sure this box is not enrolled" is idempotent. Still
		// SAID, so the operator learns the box was already inert rather than
		// inferring it from silence.
		fmt.Fprintf(stdout, "not enrolled — nothing to revoke (%s)\n", store)
		return 0
	}
	fmt.Fprintf(stdout, "revoked: the org policy plane is INERT again on this box\n")
	fmt.Fprintf(stdout, "  store: %s (removed)\n", store)
	return 0
}

func runEnrollStatus(stdout, stderr io.Writer, asJSON bool, store string) int {
	rec, enrolled, err := policy.LoadOrgEnrollment(store)
	if err != nil {
		// A damaged store is NOT reported as "not enrolled" — that is the fail-open
		// this whole path exists to close. Report the refusal and the cure.
		fmt.Fprintf(stderr, "fak enroll: %v\n", err)
		fmt.Fprintf(stderr, "  store: %s\n", store)
		fmt.Fprintf(stderr, "  next: fak enroll --revoke   (clears a damaged anchor; re-enroll afterwards)\n")
		return 1
	}
	if asJSON {
		return enrollJSON(stdout, stderr, store, rec, enrolled)
	}
	if !enrolled {
		fmt.Fprintf(stdout, "not enrolled — the org policy plane is INERT on this box\n")
		fmt.Fprintf(stdout, "  store: %s (absent)\n", store)
		fmt.Fprintf(stdout, "  next: fak enroll --org <url> --root-key <base64|file>\n")
		return 0
	}
	fmt.Fprintf(stdout, "enrolled: %s\n", rec.OrgURL)
	writeEnrollmentDetail(stdout, store, rec)
	return 0
}

// writeEnrollmentDetail renders the pinned anchor. The root key is shown as a SHA-256
// fingerprint rather than the raw blob: an operator compares an anchor against what the
// org published, and a 44-character base64 string is compared badly by eye.
func writeEnrollmentDetail(stdout io.Writer, store string, rec policy.OrgEnrollment) {
	fmt.Fprintf(stdout, "  issuer:   %s\n", rec.Issuer)
	fmt.Fprintf(stdout, "  device:   %s\n", rec.DeviceID)
	if rec.User != "" {
		fmt.Fprintf(stdout, "  user:     %s\n", rec.User)
	}
	if len(rec.Groups) > 0 {
		fmt.Fprintf(stdout, "  groups:   %s\n", strings.Join(rec.Groups, ", "))
	}
	for _, fp := range enrollKeyFingerprints(rec) {
		fmt.Fprintf(stdout, "  root key: %s\n", fp)
	}
	fmt.Fprintf(stdout, "  since:    %s\n", time.Unix(rec.EnrolledAt, 0).UTC().Format(time.RFC3339))
	fmt.Fprintf(stdout, "  store:    %s\n", store)
}

func enrollJSON(stdout, stderr io.Writer, store string, rec policy.OrgEnrollment, enrolled bool) int {
	payload := map[string]any{
		"store":    store,
		"enrolled": enrolled,
		// Always present, never null, on both arms — a consumer must not have to read
		// an absent key as "not enrolled" (#5299 house rule).
		"org_url":          rec.OrgURL,
		"issuer":           rec.Issuer,
		"device_id":        rec.DeviceID,
		"user":             rec.User,
		"groups":           append([]string{}, rec.Groups...),
		"enrolled_at":      rec.EnrolledAt,
		"key_fingerprints": enrollKeyFingerprints(rec),
	}
	if err := writeIndentedJSON(stdout, payload); err != nil {
		fmt.Fprintf(stderr, "fak enroll: %v\n", err)
		return 1
	}
	return 0
}

// enrollKeyFingerprints renders each pinned key as sha256:<hex>. A key that will not
// decode is reported AS unreadable rather than skipped: a silently-dropped anchor would
// shrink the trusted set without saying so.
func enrollKeyFingerprints(rec policy.OrgEnrollment) []string {
	keys, err := rec.RootPublicKeys()
	if err != nil {
		return []string{"unreadable (" + err.Error() + ")"}
	}
	out := make([]string, 0, len(keys))
	for _, k := range keys {
		sum := sha256.Sum256(k)
		out = append(out, "sha256:"+hex.EncodeToString(sum[:]))
	}
	return out
}

// enrollRootKey resolves --root-key into the pinned Ed25519 public key: standard base64,
// or the path of a file holding it (a 44-character blob is awkward to paste, and operators
// keep the key in a file).
//
// An absent key is a refusal, not a default. Enrolling without an anchor would write a
// store that authenticates nothing while reporting the box as enrolled — the exact
// "absence renders like a pass" shape the trust floor is built to refuse.
func enrollRootKey(value string) (ed25519.PublicKey, error) {
	v := strings.TrimSpace(value)
	if v == "" {
		return nil, fmt.Errorf("--root-key is required to enroll: pass the org root public key (base64, or a file holding it). " +
			"Fetching it from the --org URL is not implemented (#5321), and an unpinned key off the wire is the attack enrollment exists to stop")
	}
	if b, err := os.ReadFile(v); err == nil {
		v = strings.TrimSpace(string(b))
	}
	key, err := base64.StdEncoding.DecodeString(v)
	if err != nil {
		return nil, fmt.Errorf("--root-key is not base64 key material nor a readable key file: %w", err)
	}
	if len(key) != ed25519.PublicKeySize {
		return nil, fmt.Errorf("--root-key decodes to %d bytes, want a %d-byte Ed25519 public key", len(key), ed25519.PublicKeySize)
	}
	return ed25519.PublicKey(key), nil
}

// orgURLHost derives the default issuer from the org URL's host. It is a convenience, not
// a trust decision: the issuer it produces is recorded in the anchor and refused-on-change
// like every other pinned field, so a wrong default surfaces as a re-pin refusal rather
// than silently widening what may sign.
func orgURLHost(raw string) string {
	s := raw
	if i := strings.Index(s, "://"); i >= 0 {
		s = s[i+3:]
	}
	if i := strings.IndexAny(s, "/?#"); i >= 0 {
		s = s[:i]
	}
	if i := strings.LastIndex(s, "@"); i >= 0 {
		s = s[i+1:]
	}
	if i := strings.LastIndex(s, ":"); i >= 0 && !strings.Contains(s[i+1:], "]") {
		s = s[:i]
	}
	return strings.Trim(s, "[]")
}

func defaultDeviceID() string { return trimmedHostname() }

func defaultUserID() string {
	for _, k := range []string{"USER", "USERNAME", "LOGNAME"} {
		if v := strings.TrimSpace(os.Getenv(k)); v != "" {
			return v
		}
	}
	return ""
}
