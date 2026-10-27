# Setup

Takes about five minutes. You need Python 3.11 or newer and an API key for a provider that speaks the
OpenAI API format.

## 1. Install

```bash
git clone https://github.com/IrwinJam/agent-cost-harness
cd agent-cost-harness
python -m venv .venv
.venv/Scripts/activate                    # Mac or Linux: source .venv/bin/activate
python -m pip install -e .
```

## 2. Add your model's prices

The repo only has made-up example prices. Open `harness/prices.yaml` and add your provider and model,
in dollars per million tokens:

```yaml
myprovider:
  models:
    my-model:
      - {effective_from: 2026-01-01, input: 1.25, cached_input: 0.15, output: 4.25, verified: false}
```

Use the numbers from your provider's pricing page. Skip this and the first call fails with
`no price for ...`.

## 3. Set your key

```bash
export HARNESS_API_KEY=your-key            # Windows PowerShell: $env:HARNESS_API_KEY="your-key"
```

It stays in the server. Your agent never sees it.

## 4. Start the server

```bash
python -m harness serve --base-url https://api.example.com/v1 --provider myprovider --policies cache --max-per-day 2
```

Use your provider's base URL and the provider name from step 2. `--max-per-day 2` stops it at $2 a day,
so change that to what you're OK spending.

## 5. Point your agent at it

Set your agent's OpenAI base URL to:

```
http://127.0.0.1:8787/s/<conversation-key>/v1
```

Use a new key for each conversation, like `ticket-42`. The API key your agent sends can be anything.

## 6. See what happened

```bash
python -m harness spend --db runs/serve.sqlite
python -m harness profile --db runs/serve.sqlite --list
```

`spend` shows what you've spent. `profile` shows where the money went in each conversation, and where
the cache broke.

## Want to see if it's saving you money?

Add `--ab` to step 4. Half your conversations go through the harness and half don't. After 30 or so in
each half, run:

```bash
python -m harness ab --db runs/serve.sqlite
```

## Something broke?

- **`no price for ...`**: the provider or model name in `prices.yaml` doesn't match what you passed.
- **429 from the harness**: you hit one of your spending limits. Check with `spend`.
- **400 about streaming**: streaming isn't supported yet. Turn it off in your agent.
