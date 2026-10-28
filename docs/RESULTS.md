# Results so far

These are numbers from real runs on one benchmark with one model. They show what the harness found
there. They don't say what it will save on your agent.

**Setup.** Meta's muse-spark-1.3 through its OpenAI-compatible Model API, reasoning effort set to medium. The
tasks come from a public customer-service agent benchmark: a retail domain with 40 test tasks, a
simulated customer played by the same model, and automatic grading (plus a model acting as judge on 11
of the 40 tasks). The benchmark's own agent runs unchanged, with its calls routed through the harness.
Costs use Meta's published prices, which I haven't checked line by line against the billing
page yet.

Total spent on everything below: about **$30**.

## Short version

On this benchmark, `cache_layout` cut the agent's cost by **21%** (95% CI 17% to 25%), or about $21 for
every $100 the agent spends. Counting the simulated customer and judge, which it doesn't touch, the
whole bill dropped about 15%.

Success went from 94% to 92%. The test plan said it had to show success dropped by no more than 3
points, and it couldn't show that. More on why below.

Adding `reasoning` with `escalate=false` on top cut another **20 to 21%**. That held up twice: once
against runs from a different day (section 6), and again side by side on the same day (section 7),
where both setups passed exactly the same share of tasks. Together that's roughly 37% off the agent's
cost compared with no policies.

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

## 3. Pilot: reasoning effort per call

5 tasks, 1 trial each, `cache_layout` alone vs `cache_layout` plus `reasoning`. $0.80. Too small to
say anything about success, but big enough to show how the costs move.

| | cache_layout | + reasoning |
|---|---:|---:|
| Agent cost | $0.229 | $0.236 (**+3%**) |
| Reasoning tokens | 15,909 | 7,719 (-51%) |
| Output cost | $0.107 | $0.076 (-29%) |
| Uncached input cost | $0.047 | $0.075 (+60%) |
| Calls | 73 | 82 |
| Tasks passed | 5 of 5 | 4 of 5 |

**It didn't pay off on this model.** Low effort did cut reasoning in half. But the cache doesn't carry
over when the effort changes. Every time a call went from low to full effort, only the shared tools and
system prompt (5,105 tokens) came from the cache, and the rest of the conversation was paid at full
price. Going back to low got 94% to 97% from the cache right away. So this provider seems to keep a
separate cache for each effort level.

Each of the 6 escalations cost a thrown-away call plus a mostly uncached retry, and that ate the
savings. It didn't help that this model only reasoned about 217 tokens per call at full effort here,
so there wasn't much to cut.

The safety net did catch one thing. On one task the low-effort call went straight to changing an
order, and the full-effort retry asked the customer to confirm first, which the store's rules require.

## 4. Pilot: low effort for the whole conversation

Same 5 tasks, but with `escalate=false`. Every call stays at low, so the cache never switches and
nothing gets asked twice. $0.71.

| | cache_layout | + reasoning, no escalation |
|---|---:|---:|
| Agent cost | $0.219 | $0.187 (**-15%**) |
| Reasoning tokens | 15,401 | 6,956 (-55%) |
| Output cost | $0.106 | $0.069 (-35%) |
| Uncached input cost | $0.038 | $0.039 |
| Input from cache | 94.3% | 94.5% |
| Calls | 71 | 75 |
| Tasks passed | 5 of 5 | 4 of 5 |

**This one worked on cost.** Reasoning dropped by more than half, the cache held, and input cost
stayed flat, so the output savings went straight to the bottom line.

The failed task is a coin flip on this benchmark. Both setups did the right thing and cancelled the
order. The low-effort run picked the reason "ordered by mistake" instead of "no longer needed", and the
grader checks the exact reason. The same task passed only 2 of 3 times for each setup in the full test.

Still, 5 tasks can't say if success holds. Writes no longer get full effort, so a full test is the next
step before trusting this.

## 5. Live A/B check

To check `serve --ab` against the real API, I sent 64 short scripted conversations through it: a
4.7k-token store policy as the system prompt, then 3 customer questions each. The split put 34 in the
control half and 30 in the policies half (`cache_layout`). $0.55.

```
group       convos   calls    $/convo  cache hit
control         34     102     0.0110     71.2%
policies        30      90     0.0059     97.1%

cost per conversation: -46.9% (95% CI -55.0% to -36.2%)
```

That's the same thing the full test found, on separate traffic. Without a cache key, calls with an
identical prompt start kept missing the cache. With one shared key, almost every call after the first
hit it.

The saving is bigger than in the full test because these conversations are short. The fixed prompt
start is most of each call, so the cache matters more. Real agent conversations with more turns will
land closer to the 21%. And since the conversations were scripted, this checks cost only.

## 6. Low effort on all 40 tasks

The pilot couldn't say if success holds, so this ran `cache_layout` plus `reasoning` with
`escalate=false` on all 40 tasks, 2 trials each. To keep it under $7, only the low-effort setup ran.
It's compared with the `cache_layout` runs from the full test (section 1), which used the same model
and tasks at medium effort, 3 trials each. The plan, the 6-point margin and the comparison script were
all written and committed before the run. $4.64.

| | cache_layout (full test) | + low effort |
|---|---:|---:|
| Tasks passed | 91.7% | 93.8% |
| Agent cost per trial | $0.0394 | $0.0315 (**-20%**) |
| Reasoning tokens per trial | 2,936 | 1,170 (-60%) |

- **Success held.** The difference is +2.1 points (95% CI -3.3 to +8.8). The one-sided lower bound is
  -2.9 points, inside the 6-point margin, and even inside the 3-point one from the full test.
- **Cost dropped 20%** (95% CI 16% to 24%) on top of what `cache_layout` already saved.
- One task got clearly worse with low effort and one got clearly better. The better one is the
  return-and-exchange task that `cache_layout` failed 3 of 3 times in the full test.

The catch: the two setups ran on different days, with 2 trials against 3. If the provider changed
anything in between, it would show up here as a difference between the setups. A side-by-side run
would settle it. Still, this is the best sign so far that this model doesn't need medium effort for
this kind of work.

## 7. Low effort, side by side

To take the different-day question off the table, both setups ran together on all 40 tasks, 1 trial
each, taking turns going first. The plan was committed before the run. $4.96.

| | cache_layout | + low effort |
|---|---:|---:|
| Tasks passed | 92% (37 of 40) | 92% (37 of 40) |
| Agent cost per trial | $0.0391 | $0.0308 (**-21%**) |
| Reasoning tokens per trial | 2,922 | 1,212 (-59%) |
| Input from cache | 94.2% | 94.3% |

- **The saving is real.** -21% (95% CI 16% to 27%), almost exactly what the different-day run found.
  So it wasn't day-to-day drift.
- **Success was a tie,** 37 of 40 each. Each setup failed two tasks the other passed. Low effort won
  back the return-and-exchange task again.
- **But a tie on 40 single trials can't prove a small margin.** The one-sided lower bound is -7.5
  points, just past the 6-point margin, so by the plan's rule the check didn't pass. With 1 trial per
  task the uncertainty is wider than in section 6, which used 2.

Put together: two runs, 120 low-effort trials, and neither showed success dropping (+2.1 points, then
+0.0). The cost saving is solid. The success question leans clearly toward "no change", but a run with
more trials per task is what would close it formally.

## 8. What this doesn't cover

- One model, one provider, one benchmark. Your numbers will be different.
- Tool outputs here are small, and contexts stayed under about 15k tokens. `output_handles` and
  `masking` would barely do anything, so they weren't tested. They need a benchmark with long tool
  outputs.
- Prices aren't checked against the real bill yet.
