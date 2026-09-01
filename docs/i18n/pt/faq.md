---
title: "fak — Perguntas frequentes de primeiro contato (introdução em Português / Portuguese introduction)"
description: "Página de entrada em Português para o fak: um único binário Go que revisa cada tool call antes de executar e reaproveita trabalho repetido em sessões longas — o mesmo loop de agente fica mais seguro, mais barato e mais rápido. Self-host, compatível com LGPD (Brasil) / RGPD (Portugal), Apache-2.0."
---

# fak — Perguntas frequentes de primeiro contato

> Esta é uma **página de entrada localizada (entry point)**, não uma tradução completa
> da documentação. A documentação completa está em inglês — esta página dá o essencial
> do fak e encaminha você para os [documentos em inglês](https://github.com/anthony-chaudhary/fak/blob/main/README.md).
> **Aviso:** esta tradução foi gerada por máquina e ainda aguarda revisão de falante
> nativo — se encontrar erros, abra uma issue ou um PR.
>
> **Outros idiomas:** veja o [i18n hub](../README.md) para a lista completa.

## Q1. O que é o fak?

Um único **binário Go estático** que você coloca na frente do agente de IA que já
executa (Claude Code, Codex, Cursor, qualquer cliente OpenAI/Anthropic/MCP),
reapontando **uma única base URL** — sem reescrever nada. Ele torna sessões longas
mais baratas (descarta turnos antigos mantendo o prefixo do prompt-cache do provedor
byte a byte idêntico), roteia cada tool call, pode rodar modelos GGUF locais no próprio
processo e registra um veredito auditável para cada chamada.

## Q2. Preciso trocar de modelo ou reescrever meu agente?

Não. O fak **governa e faz cache do modelo que você já usa** — ele não o substitui.
Envolva seu agente com um único comando, ou reaponte uma base URL para `fak serve`:

```bash
fak manage claude      # envolve seu agente existente em um único comando
```

## Q3. Para onde vão meus dados — isto é compatível?

Self-host primeiro: um binário estático na frente de um modelo local ou de um provedor
doméstico, com residência de dados **fail-closed**, um piso de capacidades
**default-deny** e um log de auditoria à prova de adulteração para cada tool call.
Seus dados não saem da sua máquina. Isso atende à exigência de residência de dados
da **LGPD (Brasil)** e do **RGPD (Portugal)**.

## Q4. Quanto custa? É realmente grátis?

**Apache-2.0**, grátis, self-host. Sem cartão de crédito, sem fatura transfronteiriça,
sem entidade jurídica — nenhuma cobrança em reais (R$) nem em euros (€). O caminho
completo é `git clone` mais `go install`.

## Q5. Quanto mais barato ou mais rápido ele é?

Em uma sessão medida de **50 turnos x 5 agentes**, cerca de **4,1x menos trabalho** do
que uma pilha de warm-cache já **afinada** (tuned) — esse é o número honesto de
referência. A cifra de **~60x** (cerca de 19 horas para cerca de 19 minutos) vale
**apenas** contra um loop ingênuo de reenviar-tudo (naive re-send), nunca como manchete.
O ganho de reaproveitamento é **exclusivo de self-host**, para frotas com leitura
intensa (read-heavy). Como a conta de tokens dói em reais e euros enquanto os provedores
cobram em dólares, menos trabalho é uma alavanca direta de margem.

## Q6. Quais modelos funcionam?

**Qwen2/Qwen3 e GLM-MoE** são comprovadamente bit-exact no motor de referência
in-kernel. O restante (DeepSeek, Mistral, qualquer modelo de pesos abertos) entra
pela interface compatível com OpenAI: Ollama / vLLM / SGLang / llama.cpp / LM Studio
ou qualquer API compatível com OpenAI.

## Q7. Como ele barra prompt injection?

Dois portões **estruturais**, não um classificador: um piso de capacidades
**default-deny** (uma ferramenta perigosa nunca está na allow-list, não importa o que
o modelo tenha sido convencido a fazer) e **quarentena de resultados** (resultados de
ferramentas envenenados nunca chegam ao contexto do modelo). Em testes ao vivo, a
injeção alcançou a linha de base desprotegida 5/5, e o fak a bloqueou 5/5.

## Q8. Como eu instalo?

```bash
go install github.com/anthony-chaudhary/fak/cmd/fak@latest
```

Ou, a partir de um clone:

```bash
go build -o fak ./cmd/fak
```

Se o `proxy.golang.org` estiver bloqueado na sua rede, defina um `GOPROXY`
alcançável antes do `go install`.

## Prova de 60 segundos (sem key, sem modelo, sem GPU)

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection blocked, task still completes
```

## Q9. Para onde ir a seguir?

- [README (visão geral completa)](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — modelo local em 10 minutos](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [GETTING-STARTED — instale o binário](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [BENCHMARK-AUTHORITY — a origem de cada número](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — o que é shipped/simulated/stub](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)
- [Residência de dados e conformidade — para LGPD / RGPD](../../explainers/data-residency-and-compliance.md)
- [Integrations — conecte seu agente](../../integrations/README.md)

License: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE).
