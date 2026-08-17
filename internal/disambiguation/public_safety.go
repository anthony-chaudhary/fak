package disambiguation

import (
	"fmt"
	"reflect"
	"regexp"
	"strings"
)

const (
	ErrPublicSafetyLocalPath   = "DISAMBIGUATION_PUBLIC_SAFETY_LOCAL_PATH"
	ErrPublicSafetyCredential  = "DISAMBIGUATION_PUBLIC_SAFETY_CREDENTIAL"
	ErrPublicSafetyPrivateRepo = "DISAMBIGUATION_PUBLIC_SAFETY_PRIVATE_REPOSITORY"
	ErrPublicSafetyPrivateHost = "DISAMBIGUATION_PUBLIC_SAFETY_PRIVATE_HOST"
	ErrPublicSafetyControlText = "DISAMBIGUATION_PUBLIC_SAFETY_CONTROL_TEXT"
)

var (
	localPathText   = regexp.MustCompile(`(?i)(?:[a-z]:[\\/](?:users|home|work|src|tmp)[\\/]|/(?:home|users|private|var/tmp|tmp)/|file://|\\\\[^\\\s]+\\)`)
	credentialText  = regexp.MustCompile(`(?i)(?:bearer\s+[a-z0-9._~+/-]{8,}|(?:api[_-]?key|access[_-]?token|password|passwd|secret)\s*[:=]\s*\S+|\b(?:sk|ghp|github_pat)_[a-z0-9_-]{8,})`)
	privateRepoText = regexp.MustCompile(`(?i)(?:\bfak-private\b|github\.com/[a-z0-9_.-]+/[a-z0-9_.-]*private[a-z0-9_.-]*)`)
	privateHostText = regexp.MustCompile(`(?i)(?:\b[a-z0-9-]+\.(?:internal|local|lan)\b|\b(?:10\.[0-9]{1,3}(?:\.[0-9]{1,3}){2}|192\.168(?:\.[0-9]{1,3}){2}|172\.(?:1[6-9]|2[0-9]|3[01])(?:\.[0-9]{1,3}){2})\b)`)
)

// validatePublicSafety rejects material that cannot be published in a generated
// index. Reflection keeps the boundary covering future string fields by default.
func validatePublicSafety(entry Entry) error {
	return walkPublicStrings(reflect.ValueOf(entry), "entry")
}

func walkPublicStrings(value reflect.Value, field string) error {
	if value.Kind() == reflect.Pointer {
		if value.IsNil() {
			return nil
		}
		return walkPublicStrings(value.Elem(), field)
	}
	switch value.Kind() {
	case reflect.Struct:
		typ := value.Type()
		for i := 0; i < value.NumField(); i++ {
			name := typ.Field(i).Tag.Get("json")
			name = strings.Split(name, ",")[0]
			if name == "" || name == "-" {
				name = typ.Field(i).Name
			}
			if err := walkPublicStrings(value.Field(i), field+"."+name); err != nil {
				return err
			}
		}
	case reflect.Slice:
		for i := 0; i < value.Len(); i++ {
			if err := walkPublicStrings(value.Index(i), fmt.Sprintf("%s[%d]", field, i)); err != nil {
				return err
			}
		}
	case reflect.String:
		return validatePublicText(field, value.String())
	}
	return nil
}

func validatePublicText(field, text string) error {
	tests := []struct {
		code, message string
		pattern       *regexp.Regexp
	}{
		{ErrPublicSafetyCredential, "contains credential-shaped text", credentialText},
		{ErrPublicSafetyPrivateRepo, "names a private repository", privateRepoText},
		{ErrPublicSafetyLocalPath, "contains an absolute local path", localPathText},
		{ErrPublicSafetyPrivateHost, "contains a private host identity", privateHostText},
	}
	for _, test := range tests {
		if test.pattern.MatchString(text) {
			return provenanceError(test.code, field, test.message)
		}
	}
	for _, r := range text {
		if r < 0x20 && r != '\n' && r != '\t' {
			return provenanceError(ErrPublicSafetyControlText, field, "contains unsanitized control text")
		}
	}
	return nil
}
