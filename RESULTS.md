# Results so far

These are numbers from real runs on one benchmark with one model. They show what the harness found
there. They don't say what it will save on your agent.

**Setup.** A hosted reasoning model behind an OpenAI-compatible API, reasoning effort set to medium. The
tasks come from a public customer-service agent benchmark: a retail domain with 40 test tasks, a
simulated customer played by the same model, and automatic grading (plus a model acting as judge on 11
of the 40 tasks). The benchmark's own agent runs unchanged, with its calls routed through the harness.
Costs use the provider's published prices, which I haven't checked line by line against the billing
page yet.

Total spent on everything below: about **$18**.

## Short version

On this benchmark, `cache_layout` cut the agent's cost by **21%** (95% CI 17% to 25%), or about $21 for
every $100 the agent spends. Counting the simulated customer and judge, which it doesn't touch, the
whole bill dropped about 15%.

Success went from 94% to 92%. The test plan said it had to show success dropped by no more than 3
points, and it couldn't show that. More on why below.

## 1. Full test: baseline vs cache_layout

40 tasks, 3 trials each, 240 trials, with the two setups taking turns going first. The test plan, the
3-point margin included, was written and fingerprinted before the run. $17.10.

| | Baseline | cache_layout |
|---|---:|---:|
| Tasks passed | 94% | 92% |
| Passed all 3 trials | 88% | 85% |
| Agent cost per trial | $0.0502 | $0.0394 (**-21%**) |
| Customer and judge per trial | $0.0266 | $0.0262 |
| Input from cache | 83.6% | 94.4% |
| Calls per trial | 12.9 | 12.9 |
| Output tokens per trial | 4,551 | 4,661 |

**Where the savings came from.** Both setups made the same number of calls and wrote about the same
amount, so the agent didn't take shortcuts. The whole drop is input that came from the cache instead
of being paid for at full price. Uncached input went from $0.0192 to $0.0065 per trial.

The prompt start was already the same on every call in the baseline. The profiler showed the misses as
"prefix unchanged but not served from cache", which points at routing. The provider wasn't always
sending a call to a server that had the prompt start cached. `cache_layout` sends one cache key for
every conversation with the same prompt start, and that took the misses away: calls with less than
half their input cached went from 37 of 1,554 to 1 of 1,548. Since the benchmark's system prompt never
changes, the other part of `cache_layout`, moving changing lines out of the system prompt, did nothing
here.

**Why the success check failed.** Two things.

- The margin was too tight for this many tasks. With 40 tasks and 3 trials, the lower bound on the
  difference sits about 6 points under the measured difference. So even a perfect tie would have come
  out around -6 and failed a -3 margin. The plan should have used more tasks or a wider margin. I'm
  keeping the result as it is rather than moving the goalposts after the fact.
- The drop is mostly one task. On one task, baseline passed 3 of 3 and `cache_layout` passed 0 of 3.
  The customer asks for a return and an exchange together, but the store only allows one per order.
  Baseline told the customer to pick. `cache_layout` did the return first, which blocked the exchange
  the customer wanted. Two other tasks lost three trials between them, and `cache_layout` won back four
trials on three other tasks.

`cache_layout` doesn't change any words the model sees here. It sorts the tool list and adds the cache
key. So that task is either chance or the tool order nudging the model. One way to tell would be to
rerun with the cache key alone and the tool order left as is.

## 2. Earlier checks

Before the full test:

- **Direct calls ($0.04).** Both the Responses and Chat Completions formats work. Sending the same
  8.4k-token prompt start twice, the second call got 98.5% from the cache. Even at the lowest effort, a
  one-word answer used up to 318 reasoning tokens.
- **One full task ($0.11).** It passed. 81% of the agent's input came from the cache with no policies
  at all. Output was the biggest cost.
- **Pilot, 5 tasks, 1 trial each ($0.75).** `cache_layout` was 15% cheaper, but at that size the
  numbers were mostly noise. The full test is the one to go by.

## 3. What this doesn't cover

- One model, one provider, one benchmark. Your numbers will be different.
- Tool outputs here are small, and contexts stayed under about 15k tokens. `output_handles` and
  `masking` would barely do anything, so they weren't tested. They need a benchmark with long tool
  outputs.
- Prices aren't checked against the real bill yet.
