# agent-cost-harness

AI agents waste a lot of tokens by sending the same stuff over and over. This shows where an agent's
money goes, cuts the waste without breaking the prompt cache, and checks that the agent still does its
job just as well.

It works with any provider that speaks the OpenAI API format. Everything runs offline with a built-in
simulator, so you can try it without a key.

Early numbers from real runs, and what a full test would cost, are in [RESULTS.md](RESULTS.md).

## Try it

Needs Python 3.11 or newer. No key needed.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"   # Mac or Linux: .venv/bin/python
.venv/Scripts/python examples/simulate.py --clock
```

That runs one fake agent through each setup and prints something like this:

```
arm                     cost $   input tok  cache hit  breaks  top cache-break causes
baseline                0.6431     631,105        0%      19  system_prompt_changed x19
cache                   0.1291     631,148       90%       0  -
cache+handles           0.0443     175,367       91%       0  -
cache+handles+mask      0.0466      94,629       70%       4  history_rewritten x4
naive-masking           0.2263     235,487       10%      16  history_rewritten x16
```

These are simulated numbers. They show the harness works, not what you'll save. `--clock` puts a ticking
timestamp in the system prompt, so the baseline misses the cache on every call.

## What's in it

**Adapter.** Talks to any OpenAI-compatible API (Responses or Chat Completions). It retries on rate
limits. If your provider has its own rules, subclass it and add them in `check`.

**Call tracking.** Every call is saved to a local SQLite file with its tokens, cost, and the prompt split
into pieces.

**Profiler.** Shows where the money went. A tool output that got resent 12 times shows up once, with its
full cost. It also shows the exact call where the cache broke and why. It flags things that look
wasteful too, but those are leads, not proof. `runs/sim.sqlite` exists after you run the simulated
eval below.

```bash
python -m harness profile --db runs/sim.sqlite --list
python -m harness profile --db runs/sim.sqlite --run <id>
```

**Cost policies.** You can turn each one on or off.

- `cache_layout` keeps the start of the prompt the same between calls. It sorts the tools and moves
  things like timestamps out of the system prompt.
- `output_handles` shortens huge tool outputs and gives the model a `fetch_output` tool to read the rest.
  The harness runs that tool itself, so your agent never sees it.
- `masking` hides old tool outputs in batches. Hiding one every turn breaks the cache every turn and can
  cost more than doing nothing.

**Evaluation.** You write the test plan first in [eval/protocol.yaml](eval/protocol.yaml). The run
compares each setup against the baseline on the same tasks and stops at a hard spend cap. At the end you
get a short report that says if it got cheaper without getting worse, with the numbers behind it. If your
benchmark uses a model to play the user, that cost is kept separate and never counts as savings.

```bash
python -m harness eval validate
python -m harness eval estimate --tasks 40 --cost-per-trial 0.2
python -m harness eval run --simulate      # dry run with the built-in simulated benchmark
python -m harness eval certify --db runs/sim.sqlite --protocol eval/protocol.yaml
```

To test a real agent, write a `Benchmark` (see `harness/eval/benchmark.py`) that runs one task through
the session it's given and says if it passed. Then call `harness.eval.run_eval` with it.

**Simulator.** `python examples/simulate.py --clock` tests everything together without spending
anything. Its numbers aren't real savings.

## Use it from any language

`serve` runs a local server that looks like the OpenAI API and sits in front of your provider. Anything
that lets you set an OpenAI base URL can use it.

```bash
export HARNESS_API_KEY=...                 # your provider key stays here
python -m harness serve --base-url https://api.example.com/v1 --policies cache,handles,mask --max-per-day 5
python -m harness serve --base-url https://api.example.com/v1 --policies mask --set mask.keep_last=6
python -m harness serve --simulate         # no key needed
```

Point your client at `http://127.0.0.1:8787/s/<conversation-key>/v1`. Any API key works there. Use a
new key per conversation.

```bash
curl http://127.0.0.1:8787/s/demo/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "example-model", "messages": [{"role": "user", "content": "hi"}]}'
```

```ts
import OpenAI from "openai";
const client = new OpenAI({ baseURL: "http://127.0.0.1:8787/s/ticket-42/v1", apiKey: "unused" });
const reply = await client.chat.completions.create({ model: "example-model", messages: [{ role: "user", content: "hi" }] });
```

Then `python -m harness profile --db runs/serve.sqlite --list`.

It only listens on your machine. Streaming doesn't work yet.

## Spending limits

You can cap what `serve` is allowed to spend. Once a limit is hit, calls get a 429 with a message saying
which limit it was, and nothing gets sent to your provider.

```bash
python -m harness serve --base-url https://api.example.com/v1 \
  --max-total 20 --max-per-day 5 --max-per-conversation 0.50
```

- `--max-total` is everything ever spent in that trace database.
- `--max-per-day` resets at midnight, your local time.
- `--max-per-conversation` is per conversation key, so one runaway agent can't eat the whole budget.
- `--budget` only counts what this one server run spends and starts over when you restart it.

The first three read from the trace database, so they still hold after a restart. To see where you're
at:

```bash
python -m harness spend --db runs/serve.sqlite --max-total 20 --max-per-day 5
```

From Python, pass `limit=SpendLimit(total_usd=20, daily_usd=5)` to a `Session`. The limits check before
each call, so the last call before a limit can still go over by its own cost. They're only as accurate
as the prices in `prices.yaml`.

## Prices

`harness/prices.yaml` ships with made-up example prices so everything runs. Put your provider's real
prices in there, under the provider name you pass with `--provider`, before trusting any cost number.

## Tests

```bash
.venv/Scripts/python -m pytest
```

## Known gaps

- No streaming yet, and the policies don't support the Anthropic-style Messages format yet.
- Token counts per prompt piece are estimates based on characters. Totals per call are exact.
- The profiler assumes the prompt is built as tools, then system prompt, then conversation. Check that
  against your provider's real cache numbers.
- `cache_layout` moves timestamp lines into a `user` message, since not every provider allows a `system`
  message mid-chat. That could change behavior, so check success, not just cost.
