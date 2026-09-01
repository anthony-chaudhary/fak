---
title: "fak — início rápido (10 minutos até um modelo local)"
description: "Página de entrada em português do fak: do zero a uma IA local governada em cerca de 10 minutos — offline, sem chave, sem conta na nuvem, os dados ficam na sua máquina. Self-host, compatível com LGPD/RGPD, Apache-2.0."
---

# fak — início rápido (10 minutos até um modelo local)

> Esta é uma **página de entrada localizada**, não uma tradução completa da documentação.
> A documentação completa está em inglês — esta página dá o essencial do fak, a prova de
> 60 segundos e o caminho de instalação, e depois o encaminha para os
> [documentos em inglês](https://github.com/anthony-chaudhary/fak/blob/main/README.md).
> **Aviso:** esta tradução foi gerada por máquina e ainda aguarda revisão de um falante
> nativo — encontrou um erro? Abra uma issue/PR.
> **Outros idiomas:** veja o [i18n hub](../README.md).

## A promessa

Do zero a uma **IA local governada** em cerca de 10 minutos. Ela roda **offline**, não
custa nada (**sem chaves de API, sem conta na nuvem**), mantém os seus dados **na sua
máquina**, e para modelos pequenos a **CPU é suficiente** — não precisa de GPU.

## O caminho mais rápido (um comando)

Envolva o agente que você já usa com um modelo local, numa única linha:

```bash
fak manage claude
```

Você não reescreve o seu agente — o `fak` senta-se entre o agente e as ferramentas que
ele chama, revisa cada tool call *antes* de ela rodar e reutiliza o trabalho compartilhado
ao longo de uma sessão longa. O mesmo loop de agente fica **mais seguro, mais barato e mais
rápido**, sem reescrita.

## Prova de 60 segundos (sem chave, sem modelo, sem GPU)

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection blocked, task still completes
```

O piso de capacidades (default-deny) é verificado **dentro do kernel, no mesmo caminho de
chamada** — fail-closed. Uma ação que nunca esteve na lista de permissões não roda, não
importa o que convençam o modelo a pedir. E os *resultados* de ferramentas suspeitos são
mantidos em quarentena, **fora do contexto do modelo** — por estrutura, não por um detector.
Em testes ao vivo, a injeção de prompt atingiu a linha de base desprotegida 5/5; o fak a
isolou 5/5.

## O que é o fak

**fak é um único binário Go** que fica entre o seu agente de IA e as ferramentas que ele
chama. Você não troca de modelo — o fak **governa e faz cache** do modelo que você já usa.
Você aponta **uma** base URL para `fak serve`, ou envolve um agente existente com
`fak manage claude`.

**Qwen2/Qwen3 e GLM-MoE** estão provados bit-exact no motor de referência in-kernel; o restante
(DeepSeek, Mistral, qualquer modelo de pesos abertos) entra pela wire compatível com
OpenAI — Ollama / vLLM / SGLang / llama.cpp / LM Studio ou qualquer API compatível com
OpenAI.

## Quão rápido é (número honesto)

Numa sessão medida de **50 turnos × 5 agentes**, o ganho honesto sobre uma stack
*ajustada* de warm-cache é de cerca de **~4,1× menos trabalho**. O número de **~60×**
(cerca de 19 horas para cerca de 19 minutos) é verdadeiro **apenas contra o padrão ingênuo**
de reenviar tudo a cada turno — nunca o apresente como manchete. O ganho de reutilização é
**apenas em self-host** e vale para frotas com muita leitura.

O desconto de prompt-cache do provedor só sobrevive se o prefixo em cache permanecer
byte-a-byte idêntico; o fak **garante a identidade byte-a-byte do prefixo** enquanto descarta
os turnos antigos do meio. Se o provedor de fato reutiliza o cache é decisão do provedor — o
fak repassa essa chamada, não a reivindica.

## Custo, margem e conformidade

- **O custo dói em BRL e em EUR; a conta de tokens vem em dólar.** Reutilizar o trabalho
  compartilhado ao longo de sessões longas é uma alavanca direta de margem para equipes no
  Brasil e em Portugal.
- **Os dados ficam no país (LGPD no Brasil / RGPD em Portugal).** O fak é self-host primeiro:
  um binário estático na frente de um **modelo local** ou provedor doméstico, com residência
  fail-closed em cada backend, piso default-deny e log de auditoria à prova de adulteração para
  cada tool call. Os dados não saem da sua máquina.
- **Nada de atravessar fronteira de pagamento.** **Apache-2.0**, gratuito, self-host — sem
  cartão, sem fatura transfronteiriça, sem entidade legal. `git clone` mais
  `go install github.com/anthony-chaudhary/fak/cmd/fak@latest` é o caminho inteiro.

## Para onde ir agora

- [README (visão geral completa)](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — 10 minutos até um modelo local](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [GETTING-STARTED — instale o binário](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [BENCHMARK-AUTHORITY — a origem de cada número](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — o que é shipped/simulated/stub](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)
- [Residência de dados e conformidade — para LGPD / RGPD](../../explainers/data-residency-and-compliance.md)
- [Integrations — conecte o seu agente](../../integrations/README.md)

License: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE).
