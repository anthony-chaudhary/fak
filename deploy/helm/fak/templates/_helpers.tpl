{{/*
fak — shared helpers for the fak serve chart (#2665).

Every value these helpers read maps 1:1 to a real `fak serve` flag/env
(docs/fak/server-config.md) or a field of the committed manifests
(deploy/k8s/fak.yaml, deploy/k8s/overlays/gpu/fak-gpu.yaml).
*/}}

{{- define "fak.name" -}}
{{- .Chart.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "fak.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := .Values.nameOverride | default .Chart.Name -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "fak.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Selector labels. Instance-scoped so a proxy release and an inkernel release can
coexist in one namespace without cross-selecting each other's pods (the raw
manifests pin `app: fak` / `app: fak-gpu` for the same reason).
*/}}
{{- define "fak.selectorLabels" -}}
app.kubernetes.io/name: fak
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "fak.labels" -}}
helm.sh/chart: {{ include "fak.chart" . }}
{{ include "fak.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{/*
The serving mode: `proxy` (front an upstream model with --base-url) or
`inkernel` (load GGUF weights into the in-kernel engine on a GPU). Anything else
is a closed-vocabulary violation — fail, don't guess.
*/}}
{{- define "fak.mode" -}}
{{- $mode := .Values.mode | default "proxy" -}}
{{- if and (ne $mode "proxy") (ne $mode "inkernel") -}}
{{-   fail (printf "fak: values.mode must be 'proxy' or 'inkernel', got %q" $mode) -}}
{{- end -}}
{{- $mode -}}
{{- end -}}

{{/*
Cross-validate the two mode toggles so a half-set install cannot render a
half-shaped pod: inkernel ⇔ gpu.enabled=true. Call from every template.
*/}}
{{- define "fak.validate" -}}
{{- $mode := include "fak.mode" . -}}
{{- $gpu := .Values.gpu.enabled | default false -}}
{{- if and (eq $mode "inkernel") (not $gpu) -}}
{{-   fail "fak: mode=inkernel requires gpu.enabled=true (the in-kernel shape is the CUDA/GPU overlay: nvidia.com/gpu request + weights PVC)" -}}
{{- end -}}
{{- if and (eq $mode "proxy") $gpu -}}
{{-   fail "fak: gpu.enabled=true requires mode=inkernel (proxy mode has no GPU work)" -}}
{{- end -}}
{{- end -}}

{{/*
The image: static tag in proxy mode, the -cuda variant in inkernel mode
(Dockerfile.cuda, #2661).
*/}}
{{- define "fak.image" -}}
{{- if eq (include "fak.mode" .) "inkernel" -}}
{{- printf "%s:%s" .Values.image.repository (.Values.image.cudaTag | default "cuda-latest") -}}
{{- else -}}
{{- printf "%s:%s" .Values.image.repository (.Values.image.tag | default "0.36.0") -}}
{{- end -}}
{{- end -}}

{{/*
The Secret the env secretKeyRefs point at — <fullname>-secrets, mirroring the
committed manifests' fak-secrets / fak-gpu-secrets.
*/}}
{{- define "fak.secretsName" -}}
{{- printf "%s-secrets" (include "fak.fullname" .) -}}
{{- end -}}

{{/*
The provider key's entry inside the Secret: apiKeyEnv lowercased, underscores to
dashes — OPENAI_API_KEY becomes openai-api-key, exactly the committed manifest's
key.
*/}}
{{- define "fak.providerSecretKey" -}}
{{- regexReplaceAll "_" (lower .Values.apiKeyEnv) "-" -}}
{{- end -}}

{{/*
The policy ConfigMap name: an operator-supplied existing ConfigMap wins;
otherwise the chart creates <fullname>-policy from values.policy.json.
*/}}
{{- define "fak.policyConfigMapName" -}}
{{- if .Values.policy.configMapName -}}
{{- .Values.policy.configMapName -}}
{{- else -}}
{{- printf "%s-policy" (include "fak.fullname" .) -}}
{{- end -}}
{{- end -}}
