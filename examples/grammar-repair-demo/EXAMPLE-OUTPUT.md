# Captured output

```text
$ go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool create_support_ticket --grammar-schema examples/grammar-repair-demo/create-support-ticket.schema.json --args '{"_positional":["please help me"]}' --show-dispatched-args
verdict=TRANSFORM reason=MISROUTE by=grammar
dispatched_args={"body":"please help me"}

$ go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool create_support_ticket --grammar-schema examples/grammar-repair-demo/create-support-ticket.schema.json --args '{"_positional":["a","b","c"]}' --show-dispatched-args
verdict=DENY reason=MISROUTE by=grammar
```
