---
title: "fak クイックスタート — 約 10 分でローカルモデルを動かす（日本語 / Japanese quickstart）"
description: "fak の日本語クイックスタート：ゼロから約 10 分で、ガバナンス付きのローカル AI を——オフライン、キー不要、クラウド課金ゼロ、データは自分のマシンに残る。1 コマンドで既存エージェントをローカルモデルで包み、60 秒でツール呼び出しの境界を実証。"
---

# fak クイックスタート — 約 10 分でローカルモデルを動かす

> これは**ローカライズされた入口ページ (entry point)** であり、ドキュメントの完全な翻訳では
> ありません。正式なドキュメントは英語です。このページは機械生成であり、ネイティブレビュー
> 待ちです——修正は issue/PR で歓迎します。
> 全言語の入口は [i18n ハブ](../README.md) から、信頼できる原典（source of truth）は
> [英語版 README](https://github.com/anthony-chaudhary/fak/blob/main/README.md) をご覧ください。
>
> 同じ言語の関連ページ：[日本語版イントロダクション](./README.md)。

## このページで得られるもの

ゼロから約 **10 分**で、ガバナンス付きのローカル AI が手元で動きます——**オフライン**で動作し、
**API キー不要**、**クラウド課金（円建ての請求）ゼロ**、**データは自分のマシンから出ません**。

## 最速の道：既存エージェントをローカルモデルで包む

エージェントを書き換える必要はありません。次の 1 コマンドで、いま使っているエージェントを
ローカルモデルの前段に置き、すべての tool call を capability floor（既定拒否の権限フロア）に
通します。

```bash
fak manage claude      # 既存のエージェントをローカルモデルで包む（1 コマンド）
```

base URL を `fak serve` に向け直すだけでも同じ経路を通せます。どちらの場合も、同じエージェント
ループが、他に何も変えることなく**より安全に、より安く、より速く**なります。

## 60 秒で確かめる（キー不要・モデル不要・GPU 不要）

モデルのダウンロードも API キーも GPU もいりません。ツール呼び出しの境界だけを構造的に実証
できます——`refund_payment` は既定拒否で **DENY**、`search_kb` は **ALLOW**、注入は阻止されても
タスクは完了します。

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection blocked, task still completes
```

DENY は分類器（classifier）ではなく**構造**によるものです。許可リストに載っていない操作は、
モデルが何を言いくるめられても実行できません（fail-closed）。疑わしいツールの*結果*は
quarantine（隔離）に入れられ、そもそもモデルのコンテキストに到達しません。ライブテストでは、
未保護のベースラインに prompt injection が 5/5 到達したのに対し、fak は 5/5 で遮断しました。

## fak とは何か

**fak は 1 個の静的 Go バイナリ**で、AI エージェントとそれが呼び出すツールの間に入ります。
すべての tool call を*実行する前に*審査し、長いセッションでは繰り返される共有作業を再利用します。
結果として、同じエージェントループが、書き換えなしで安全・安価・高速になります。self-host
ファーストなので、データは自社インフラに残ります（日本のデータレジデンシー——**個人情報保護法
（APPI）** を意識したフリートに向く性質であって、法的助言ではありません）。

## どれくらい速いのか（誠実な数字）

- **誠実な見出しの数字：約 4.1×。** 50 ターン × 5 エージェントの実測セッションで、
  **チューニング済みの warm-cache スタック**より約 **4.1× 少ない作業**で済みます。トークン課金は
  円建てのコスト——これはそのままマージンのレバーになります。
- **約 60× は「naive パターンに対してのみ」、チューニング済み warm-cache スタックに対しては約 4.1×。** 毎ターン会話全体を再送する **naive な再送ループ**（約 19 時間）に対しては同じセッションが約 19 分（約 60×）まで下がりますが、この数字は naive パターンとの比較でのみ成り立ちます。見出しには使いません。
- この再利用の効果は **self-host のみ**で、read-heavy なフリートに当てはまります。
- provider の prompt-cache 割引は、cached prefix がバイト単位で同一である間だけ保たれます。
  fak は途中の古いターンを捨てつつ prefix を**バイト単位で同一に保つことを保証**します。
  provider が実際にキャッシュを再利用するかは provider 側の判断であり、fak はそれを主張するのでは
  なく中継します。

数字はすべて
[BENCHMARK-AUTHORITY](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
で commit と成果物まで辿れます。

## モデルについて

fak はモデルを置き換えません——モデルを govern して cache します。**Qwen2/Qwen3 と GLM-MoE**
は in-kernel のリファレンスエンジンで bit-exact（逐ビット一致）が実証済みです。それ以外
（DeepSeek・Mistral をはじめ、あらゆるオープンウェイトのモデル）は OpenAI 互換ワイヤ経由で
前段に置けます——Ollama / vLLM / SGLang / llama.cpp / LM Studio または任意の OpenAI 互換 API。

## ライセンスと導入コスト

fak は **Apache-2.0**、無料、self-host です——クレジットカードも、越境インボイスも、法人格も
不要。`git clone` と
`go install github.com/anthony-chaudhary/fak/cmd/fak@latest` が導入の全経路です。

## 次に読むもの

- [README（全体像）](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — 10 分でローカルモデルを動かす](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [Getting Started — バイナリをインストール](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [BENCHMARK-AUTHORITY — すべての数字の出典](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — shipped/simulated/stub の区別](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)
- [Data residency & compliance — 個人情報保護法（APPI）を意識したデータレジデンシー](../../explainers/data-residency-and-compliance.md)
- [Integrations — エージェントを接続](../../integrations/README.md)
- [日本語版イントロダクション](./README.md) ・ [i18n ハブ](../README.md)

ライセンス：[Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE)。
