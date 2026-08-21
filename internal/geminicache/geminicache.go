// Package geminicache adapts Google's explicit CachedContent lifecycle into
// fak-visible, provider-witnessed cache state.
package geminicache

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

const ProvenanceSchema = "fak/gemini-cached-content/v1"

type Route string

const (
	RouteGenerateContent Route = "generateContent"
	RouteInteractions    Route = "interactions"
)

type Identity struct {
	Account      string `json:"account"`
	Project      string `json:"project,omitempty"`
	Location     string `json:"location,omitempty"`
	Model        string `json:"model"`
	PrefixDigest string `json:"prefix_digest"`
}

func NewIdentity(account, project, location, model string, prefix []byte) Identity {
	sum := sha256.Sum256(prefix)
	return Identity{Account: account, Project: project, Location: location, Model: model, PrefixDigest: hex.EncodeToString(sum[:])}
}

func (id Identity) validate() error {
	if id.Account == "" || id.Model == "" || len(id.PrefixDigest) != 64 {
		return errors.New("gemini cache: incomplete account/model/prefix identity")
	}
	return nil
}

type Admission struct {
	PredictedReuseValueUSD float64       `json:"predicted_reuse_value_usd"`
	CreationStorageCostUSD float64       `json:"creation_storage_cost_usd"`
	TTL                    time.Duration `json:"ttl"`
	MaxTTL                 time.Duration `json:"max_ttl"`
	PrivacyAllowed         bool          `json:"privacy_allowed"`
}

func (a Admission) Check() error {
	if !a.PrivacyAllowed {
		return errors.New("gemini cache: privacy ceiling refuses provider residency")
	}
	if a.TTL <= 0 || (a.MaxTTL > 0 && a.TTL > a.MaxTTL) {
		return errors.New("gemini cache: TTL exceeds residency ceiling")
	}
	if a.PredictedReuseValueUSD <= a.CreationStorageCostUSD {
		return errors.New("gemini cache: predicted reuse value does not exceed creation/storage cost")
	}
	return nil
}

type Content struct {
	Role  string `json:"role,omitempty"`
	Parts []Part `json:"parts"`
}

type Part struct {
	Text string `json:"text,omitempty"`
}

type CreateConfig struct {
	DisplayName string    `json:"displayName,omitempty"`
	Contents    []Content `json:"contents,omitempty"`
	TTL         string    `json:"ttl,omitempty"`
	ExpireTime  time.Time `json:"expireTime,omitempty"`
}

type UpdateConfig struct {
	TTL        string    `json:"ttl,omitempty"`
	ExpireTime time.Time `json:"expireTime,omitempty"`
}

type UsageMetadata struct {
	TotalTokenCount int64 `json:"totalTokenCount,omitempty"`
}

type CachedContent struct {
	Name          string         `json:"name,omitempty"`
	DisplayName   string         `json:"displayName,omitempty"`
	Model         string         `json:"model,omitempty"`
	CreateTime    time.Time      `json:"createTime,omitempty"`
	UpdateTime    time.Time      `json:"updateTime,omitempty"`
	ExpireTime    time.Time      `json:"expireTime,omitempty"`
	UsageMetadata *UsageMetadata `json:"usageMetadata,omitempty"`
}

type State string

const (
	StateActive  State = "active"
	StateExpired State = "expired"
	StateDeleted State = "deleted"
)

type Receipt struct {
	Schema   string          `json:"schema"`
	Identity Identity        `json:"identity"`
	Object   CachedContent   `json:"object"`
	State    State           `json:"state"`
	Observed time.Time       `json:"observed_at"`
	Raw      json.RawMessage `json:"raw"`
}

type UnsupportedError struct {
	Route  Route
	Reason string
}

func (e *UnsupportedError) Error() string {
	return fmt.Sprintf("gemini cache: %s unsupported: %s", e.Route, e.Reason)
}

type Capabilities struct {
	GenerateContent bool
	Models          map[string]bool
	Locations       map[string]bool
}

type Client struct {
	BaseURL      string
	APIKey       string
	HTTPClient   *http.Client
	Now          func() time.Time
	Capabilities Capabilities
}

func (c Client) requireExplicitCache(route Route, id Identity) error {
	if route != RouteGenerateContent {
		return &UnsupportedError{Route: route, Reason: "explicit CachedContent objects require generateContent"}
	}
	if !c.Capabilities.GenerateContent {
		return &UnsupportedError{Route: route, Reason: "generateContent CachedContent capability was not negotiated"}
	}
	if len(c.Capabilities.Models) > 0 && !c.Capabilities.Models[id.Model] {
		return &UnsupportedError{Route: route, Reason: "model does not advertise explicit caching"}
	}
	if len(c.Capabilities.Locations) > 0 && !c.Capabilities.Locations[id.Location] {
		return &UnsupportedError{Route: route, Reason: "location does not advertise explicit caching"}
	}
	return nil
}

func validateName(name string) error {
	parts := strings.Split(name, "/")
	if len(parts) != 2 || parts[0] != "cachedContents" || strings.TrimSpace(parts[1]) == "" {
		return errors.New("gemini cache: invalid provider object name")
	}
	return nil
}

func (c Client) endpoint(path string) (string, error) {
	base := strings.TrimRight(c.BaseURL, "/")
	if base == "" {
		base = "https://generativelanguage.googleapis.com/v1beta"
	}
	u, err := url.Parse(base + "/" + strings.TrimLeft(path, "/"))
	if err != nil {
		return "", err
	}
	if c.APIKey != "" {
		q := u.Query()
		q.Set("key", c.APIKey)
		u.RawQuery = q.Encode()
	}
	return u.String(), nil
}

func (c Client) do(ctx context.Context, method, path string, body any, out any) (json.RawMessage, error) {
	endpoint, err := c.endpoint(path)
	if err != nil {
		return nil, err
	}
	var reader io.Reader
	if body != nil {
		encoded, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		reader = bytes.NewReader(encoded)
	}
	req, err := http.NewRequestWithContext(ctx, method, endpoint, reader)
	if err != nil {
		return nil, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	hc := c.HTTPClient
	if hc == nil {
		hc = http.DefaultClient
	}
	resp, err := hc.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("gemini cache: %s %s returned %s: %s", method, path, resp.Status, strings.TrimSpace(string(raw)))
	}
	if out != nil && len(bytes.TrimSpace(raw)) > 0 {
		if err := json.Unmarshal(raw, out); err != nil {
			return nil, err
		}
	}
	return append(json.RawMessage(nil), raw...), nil
}

func (c Client) receipt(id Identity, object CachedContent, raw json.RawMessage, forced State) Receipt {
	now := time.Now().UTC()
	if c.Now != nil {
		now = c.Now().UTC()
	}
	state := forced
	if state == "" {
		state = StateActive
		if !object.ExpireTime.IsZero() && !object.ExpireTime.After(now) {
			state = StateExpired
		}
	}
	return Receipt{Schema: ProvenanceSchema, Identity: id, Object: object, State: state, Observed: now, Raw: raw}
}

func (c Client) Create(ctx context.Context, route Route, id Identity, admission Admission, config CreateConfig) (Receipt, error) {
	if err := c.requireExplicitCache(route, id); err != nil {
		return Receipt{}, err
	}
	if err := id.validate(); err != nil {
		return Receipt{}, err
	}
	if err := admission.Check(); err != nil {
		return Receipt{}, err
	}
	config.TTL = durationString(admission.TTL)
	body := struct {
		Model string `json:"model"`
		CreateConfig
	}{Model: id.Model, CreateConfig: config}
	var object CachedContent
	raw, err := c.do(ctx, http.MethodPost, "cachedContents", body, &object)
	if err != nil {
		return Receipt{}, err
	}
	if object.Model != "" && object.Model != id.Model {
		return Receipt{}, errors.New("gemini cache: provider returned a different model identity")
	}
	return c.receipt(id, object, raw, ""), nil
}

func (c Client) Get(ctx context.Context, id Identity, name string) (Receipt, error) {
	if err := id.validate(); err != nil {
		return Receipt{}, err
	}
	if err := validateName(name); err != nil {
		return Receipt{}, err
	}
	var object CachedContent
	raw, err := c.do(ctx, http.MethodGet, name, nil, &object)
	if err != nil {
		return Receipt{}, err
	}
	if object.Name != name || (object.Model != "" && object.Model != id.Model) {
		return Receipt{}, errors.New("gemini cache: observed object identity does not match reference")
	}
	return c.receipt(id, object, raw, ""), nil
}

func (c Client) Update(ctx context.Context, id Identity, name string, config UpdateConfig) (Receipt, error) {
	if err := id.validate(); err != nil {
		return Receipt{}, err
	}
	if err := validateName(name); err != nil {
		return Receipt{}, err
	}
	var object CachedContent
	raw, err := c.do(ctx, http.MethodPatch, name, config, &object)
	if err != nil {
		return Receipt{}, err
	}
	if object.Name != name || (object.Model != "" && object.Model != id.Model) {
		return Receipt{}, errors.New("gemini cache: updated object identity does not match reference")
	}
	return c.receipt(id, object, raw, ""), nil
}

func (c Client) Delete(ctx context.Context, id Identity, name string) (Receipt, error) {
	if err := id.validate(); err != nil {
		return Receipt{}, err
	}
	if err := validateName(name); err != nil {
		return Receipt{}, err
	}
	raw, err := c.do(ctx, http.MethodDelete, name, nil, nil)
	if err != nil {
		return Receipt{}, err
	}
	return c.receipt(id, CachedContent{Name: name, Model: id.Model}, raw, StateDeleted), nil
}

func (c Client) List(ctx context.Context, pageSize int, pageToken string) ([]CachedContent, string, json.RawMessage, error) {
	path := fmt.Sprintf("cachedContents?pageSize=%d", pageSize)
	if pageToken != "" {
		path += "&pageToken=" + url.QueryEscape(pageToken)
	}
	var response struct {
		CachedContents []CachedContent `json:"cachedContents"`
		NextPageToken  string          `json:"nextPageToken"`
	}
	raw, err := c.do(ctx, http.MethodGet, path, nil, &response)
	return response.CachedContents, response.NextPageToken, raw, err
}

func Reference(route Route, expected Identity, receipt Receipt, prefix []byte, now time.Time) (string, error) {
	if route != RouteGenerateContent {
		return "", &UnsupportedError{Route: route, Reason: "explicit CachedContent objects require generateContent"}
	}
	actual := NewIdentity(expected.Account, expected.Project, expected.Location, expected.Model, prefix)
	if actual != expected || receipt.Identity != expected {
		return "", errors.New("gemini cache: account/project/location/model/prefix identity mismatch")
	}
	if receipt.State != StateActive || (!receipt.Object.ExpireTime.IsZero() && !receipt.Object.ExpireTime.After(now)) {
		return "", errors.New("gemini cache: object is expired or deleted")
	}
	if receipt.Object.Name == "" || (receipt.Object.Model != "" && receipt.Object.Model != expected.Model) {
		return "", errors.New("gemini cache: provider object identity mismatch")
	}
	return receipt.Object.Name, nil
}

func durationString(d time.Duration) string {
	seconds := d / time.Second
	return fmt.Sprintf("%ds", seconds)
}
