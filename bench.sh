#!/usr/bin/env bash
# GuideLLM benchmark against your public ZeroHost endpoint (Linux, macOS or WSL).
#   bash bench.sh [prompt_tokens] [output_tokens] [profile] [seconds_per_level] [extra guidellm args...]
#   bash bench.sh 256 64 synchronous 30                 # single user, best case
#   bash bench.sh 512 128 sweep 60                      # automatic load sweep
#   bash bench.sh 512 128 constant 60 --override profile.rate 0.8,1.0,1.2,1.4
# Reads NGROK_DOMAIN and VLLM_API_KEY from zerohost-secrets/secrets.json.
# Reports land in results/<timestamp>_p<prompt>_o<output>_<profile>.{json,html,csv}
# Run one benchmark at a time: overlapping runs share the GPU and corrupt each other's numbers.
set -euo pipefail
cd "$(dirname "$0")"

PROMPT_TOKENS=${1:-512}
OUTPUT_TOKENS=${2:-128}
PROFILE=${3:-sweep}
SECONDS_PER_LEVEL=${4:-60}
shift $(( $# < 4 ? $# : 4 ))

secret() { python3 -c "import json;print(json.load(open('zerohost-secrets/secrets.json'))['$1'])"; }
URL="https://$(secret NGROK_DOMAIN)"
API_KEY=$(secret VLLM_API_KEY)
GUIDELLM=$(command -v guidellm || echo ~/.local/bin/guidellm)

name="results/$(date +%Y%m%d_%H%M%S)_p${PROMPT_TOKENS}_o${OUTPUT_TOKENS}_${PROFILE}"
mkdir -p results

"$GUIDELLM" run \
  --backend "{\"kind\":\"openai_http\",\"target\":\"$URL\",\"model\":\"qwen2.5-7b-instruct\",\"api_key\":\"$API_KEY\",\"extras\":{\"headers\":{\"ngrok-skip-browser-warning\":\"1\"}}}" \
  --tokenizer kind=hf_auto,model=Qwen/Qwen2.5-7B-Instruct-AWQ \
  --data "kind=synthetic_text,prompt_tokens=$PROMPT_TOKENS,output_tokens=$OUTPUT_TOKENS" \
  --profile "kind=$PROFILE" \
  --constraint "kind=max_duration,seconds=$SECONDS_PER_LEVEL" \
  --output "kind=json,path=$name.json" \
  --output "kind=html,path=$name.html" \
  --output "kind=csv,path=$name.csv" \
  --output kind=console \
  "$@"
