---
title: "fak — instalação e primeiros passos (introdução em Português / Portuguese introduction)"
description: "Página de entrada em Português para instalar e executar fak: uma única binária Go que revisa cada tool call antes de rodar e reutiliza trabalho compartilhado — o mesmo loop de agente, mais seguro, mais barato e mais rápido. Self-host, compatível com LGPD/RGPD, Apache-2.0."
---

# fak — instalação e primeiros passos

> Esta é uma **página de entrada localizada (entry point)**, não uma tradução completa da documentação.
> A documentação completa está em inglês — esta página dá a você o caminho de instalação e o mapa de
> tiers, e depois o encaminha para a [documentação em inglês](https://github.com/anthony-chaudhary/fak/blob/main/README.md).
> **Aviso:** esta tradução foi gerada por máquina e ainda aguarda revisão de falante nativo — se
> encontrar um erro, abra uma issue/PR.
>
> Outros idiomas e o índice de traduções estão no [i18n hub](../README.md).

Esta é a porta de entrada de **instalar e executar**; o pitch denso está no
[README](https://github.com/anthony-chaudhary/fak/blob/main/README.md). `fak` é **uma única binária Go**,
estática, sem dependências externas: a mesma binária é o gateway, o motor de KV-cache e roteamento, e —
no mesmo caminho de chamada — o portão de política, a quarentena de resultados e o log de auditoria.
Você não reescreve seu agente: aponta uma base URL para `fak serve`, ou envolve um agente existente com
um único comando, `fak manage -- claude`.

## Por que importa no Brasil e em Portugal

- **Os dados permanecem no seu ambiente (LGPD no Brasil / RGPD em Portugal).** fak é self-host primeiro:
  uma binária estática que se posiciona à frente de um **modelo local** ou de um provider, com residência
  de dados fail-closed em cada backend, um capability floor default-deny e um log de auditoria de cada
  tool call. Os dados não saem da sua máquina.
- **Custo e margem em BRL e EUR.** Em sessões longas, fak reutiliza o trabalho compartilhado (o KV cache
  do system prompt e da lista de ferramentas): **cerca de 4,1× menos trabalho** que um stack warm-cache já
  ajustado, medido em uma sessão de 50 turnos × 5 agentes. (A cifra de ~60×, de ~19 horas para ~19 minutos,
  vale **apenas** contra o padrão ingênuo de reenviar tudo a cada turno — nunca use ~60× como manchete.)
  Esse ganho de reutilização é **exclusivo do self-host** e se aplica a frotas com leitura intensiva.
- **Sem atravessar meios de pagamento.** fak é **Apache-2.0**, gratuito, self-host — sem cartão de crédito,
  sem fatura transfronteiriça, sem entidade legal. `git clone` mais `go install` é o caminho inteiro.

## 0. Pré-requisitos

- **Go 1.26+.** O `go.mod` declara `go 1.26`; com o `GOTOOLCHAIN=auto` padrão do Go, um `go` mais antigo
  baixa o toolchain correto automaticamente na primeira build (precisa de rede uma vez). Verifique com
  `go version`.
- **Para o Tier 0, é só isso**: sem GPU, sem API key, sem rede.

## 1. Mapa de tiers

Há quatro coisas que você pode fazer, em ordem crescente de custo de setup — e **nada novo é instalado
entre elas**:

| Tier | O que você obtém | Setup |
|---|---|---|
| **0 — Experimentar o kernel** | Rodar/medir o limite de adjudicação, offline | `go build` |
| **1 — Colocar fak à frente de um modelo real** | O kernel à frente de um modelo que você serve em outro lugar (Ollama / vLLM / llama.cpp / cloud) | + um servidor compatível com OpenAI |
| **1b — Modelo local em um comando** | Um modelo GGUF local, in-kernel, com seu agente atual — sem key, sem rede, sem segundo terminal | `fak manage --gguf qwen2.5:7b -- claude` |
| **2 — O modelo fundido no kernel** | O forward pass em Go puro que o próprio kernel executa | + export das weights reais |

Se você só quer **servir um modelo útil com fak à frente**, o alvo é o **Tier 1**. O modelo in-kernel do
Tier 2 é um *forward pass de referência*, comprovado bit a bit contra o HuggingFace — não um motor de
serviço com qualidade de chat.

## 2. Instalar a binária

Com Go, o module path é a raiz do repositório, então instala direto:

```bash
go install github.com/anthony-chaudhary/fak/cmd/fak@latest   # -> $(go env GOBIN) / $GOPATH/bin
```

A partir do clone (contribuidor):

```bash
git clone https://github.com/anthony-chaudhary/fak.git
cd fak
go build -o fak ./cmd/fak          # -> ./fak   (Windows: build com -o fak.exe)
./fak help
```

## 3. Prova em 60 segundos (sem key, sem modelo, sem GPU)

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection blocked, task still completes
```

O capability floor é **default-deny, verificado dentro do kernel no mesmo caminho de chamada** (fail-closed):
uma ação que nunca está na allow-list não roda, não importa o que convenceram o modelo a fazer. Resultados
suspeitos de ferramentas ficam em **quarentena**, fora do contexto do modelo — por estrutura, não por um
detector. Em testes ao vivo, a prompt injection atingiu a baseline desprotegida 5/5; fak a bloqueou 5/5.

## 4. Com o seu modelo

fak governa e faz cache do seu modelo; não o substitui. **Qwen2/Qwen3 e GLM-MoE** são comprovadamente
bit-exact no motor de referência in-kernel. O restante (DeepSeek, Mistral, qualquer modelo de open-weights)
conecta pela wire compatível com OpenAI: Ollama / vLLM / SGLang / llama.cpp / LM Studio ou qualquer API
compatível com OpenAI.

Sobre o prompt-cache do provider: fak mantém o prefixo em cache byte a byte idêntico enquanto descarta os
turnos antigos do meio, de modo que o desconto do provider sobrevive. fak **garante** a identidade byte a byte
do prefixo; se o provider reutiliza o cache é decisão do provider, que fak repassa em vez de afirmar.

## Para onde ir em seguida

- [README — visão geral completa](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — modelo local em 10 minutos](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [GETTING-STARTED — instalar a binária, mapa de tiers, tutorial guiado](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [BENCHMARK-AUTHORITY — a origem de cada número](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — o que é shipped/simulated/stub](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)
- [Residência de dados e conformidade — para LGPD / RGPD](../../explainers/data-residency-and-compliance.md)
- [Integrations — conecte o seu agente](../../integrations/README.md)

License: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE).
