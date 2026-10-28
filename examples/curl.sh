#!/usr/bin/env sh
# Send one conversation through the harness with nothing but curl.
#
#   python -m harness serve --simulate      (in another terminal)
#   sh examples/curl.sh
#
# Then look at it with: python -m harness profile --db runs/serve.sqlite --list

HARNESS_URL="${HARNESS_URL:-http://127.0.0.1:8787}"
KEY="curl-$(date +%s)"

ask() {
  curl -s "$HARNESS_URL/s/$KEY/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"${MODEL:-example-model}\", \"messages\": [
          {\"role\": \"system\", \"content\": \"You help customers with their orders.\"},
          {\"role\": \"user\", \"content\": \"$1\"}]}"
  echo
}

ask "Where is my order?"
ask "Can I change the shipping address?"
echo "conversation key: $KEY"
