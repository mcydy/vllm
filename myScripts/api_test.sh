#!/usr/bin/env bash
# vLLM OpenAI 接口全覆盖验证
# 用法: sh api_test.sh BASE=http://192.168.1.10:8000 bash vllm_api_check.sh

BASE="${BASE:-http://127.0.0.1:8000}"
JSON='Content-Type: application/json'

C_G='\033[32m'; C_R='\033[31m'; C_Y='\033[33m'; C_B='\033[36m'; C_0='\033[0m'

hr() { printf '\n\033[35m══════ %s ══════\033[0m\n' "$1"; }

# 通用请求：$1=方法 $2=路径 $3=body（GET 时留空）
req() {
  local m="$1" p="$2" b="${3:-}"
  printf '%b[%s] %s%b\n' "$C_B" "$m" "$p" "$C_0"
  if [ "$m" = "GET" ]; then
    curl -s -w '\n%{http_code}' "$BASE$p"
  else
    curl -s -w '\n%{http_code}' -X POST "$BASE$p" -H "$JSON" -d "$b"
  fi
}

# 摘取返回体关键字段
peek() {
  local r="$1"
  echo "$r" | head -c 400
  echo "$r" | grep -oE '"(id|object|model|finish_reason)": *"[^"]*"' | head -4
  echo "$r" | grep -oE '"content": *"[^"]{0,80}' | head -2
  echo "$r" | grep -oE '"text": *"[^"]{0,80}' | head -2
  echo "$r" | grep -oE '"(total_tokens|prompt_tokens|completion_tokens)": *[0-9]+' | head -3
}

echo "目标: $BASE"

# ── 0. 连通性 + 抓取真实模型名 ────────────────────────
hr "0. 连通性与模型发现"

code=$(curl -s -o /dev/null -w '%{http_code}' "$BASE/health")
echo -n "GET /health            → "
[ "$code" = "200" ] && printf '%b%s OK%b\n' "$C_G" "$code" "$C_0" \
                    || printf '%b%s (未就绪/异常)%b\n' "$C_Y" "$code" "$C_0"

MODELS=$(curl -s "$BASE/v1/models")
MODEL=$(echo "$MODELS" | grep -oE '"id": *"[^"]+"' | head -1 | sed 's/.*: *"//;s/"//')
[ -z "$MODEL" ] && MODEL=$(echo "$MODELS" | grep -oE '"(model|id)": *"[^"]+"' | head -1 | sed 's/.*: *"//;s/"//')

if [ -z "$MODEL" ]; then
  printf '%b拿不到模型名，后续测试大概率 404。原始返回：%b\n' "$C_R" "$C_0"
  echo "$MODELS" | head -c 300
  exit 1
fi
printf '自动探测到模型名: %b%s%b\n\n' "$C_G" "$MODEL" "$C_0"
echo "$MODELS" | head -c 300

mk() { printf '{"model":"%s","messages":[{"role":"user","content":"%s"}],"max_tokens":32}' "$MODEL" "$1"; }

# ── 1. 只读端点 ──────────────────────────────────────
hr "1. 只读端点"
for p in /version /load /is_scaling_elastic_ep; do
  r=$(req GET "$p"); printf '  HTTP %s  ' "$(echo "$r"|tail -1)"; echo "$r"|head -1|head -c 150; echo
done

# ── 2. Tokenizer ─────────────────────────────────────
hr "2. Tokenizer（查 prompt 实际 token 数）"
r=$(req POST /tokenize "{\"model\":\"$MODEL\",\"prompt\":\"请详细介绍一下北京这座城市的历史文化和名胜古迹\"}")
echo "$r" | head -1 | head -c 200
echo "  ↑ tokens 数量决定 prefix caching 能否命中（block_size 默认 16）"

r=$(req POST /detokenize "{\"model\":\"$MODEL\",\"tokens\":[9707,11,358]}")
echo "$r" | head -1 | head -c 200

# ── 3. 核心：completions / chat ──────────────────────
hr "3. 核心生成接口"

r=$(req POST /v1/completions "{\"model\":\"$MODEL\",\"prompt\":\"北京是\",\"max_tokens\":32,\"temperature\":0.7}")
printf '  HTTP %s\n' "$(echo "$r"|tail -1)"; peek "$r"

r=$(req POST /v1/chat/completions "$(mk '用一句话介绍北京')")
printf '  HTTP %s\n' "$(echo "$r"|tail -1)"; peek "$r"

# ── 4. 流式 ──────────────────────────────────────────
hr "4. 流式 SSE"
echo "  发送 stream=true，取前 6 个 chunk："
curl -s -N -X POST "$BASE/v1/chat/completions" -H "$JSON" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"数到5\"}],\"max_tokens\":32,\"stream\":true}" \
  | head -6 | sed 's/^/    /'

# ── 5. 采样参数 ──────────────────────────────────────
hr "5. 采样参数 / 停止条件"
r=$(req POST /v1/chat/completions "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"写一首短诗\"}],\"max_tokens\":64,\"temperature\":0.9,\"top_p\":0.95,\"top_k\":40,\"stop\":[\"\\n\\n\"],\"seed\":42}")
printf '  HTTP %s\n' "$(echo "$r"|tail -1)"; peek "$r"

# ── 6. logprobs（vLLM 扩展） ────────────────────────
hr "6. logprobs（vLLM 扩展字段）"
r=$(req POST /v1/chat/completions "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"北京是\"}],\"max_tokens\":5,\"logprobs\":true,\"top_logprobs\":3}")
echo "$r" | head -1 | head -c 300

# ── 7. Anthropic 兼容 ────────────────────────────────
hr "7. Anthropic 兼容 /v1/messages"
r=$(req POST /v1/messages "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}],\"max_tokens\":32}")
printf '  HTTP %s\n' "$(echo "$r"|tail -1)"; peek "$r"

r=$(req POST /v1/messages/count_tokens "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"你好世界\"}]}")
printf '  HTTP %s\n' "$(echo "$r"|tail -1)"; echo "$r"|head -1|head -c 150

# ── 8. Responses API ─────────────────────────────────
hr "8. Responses API（用 input 不是 messages）"
r=$(req POST /v1/responses "{\"model\":\"$MODEL\",\"input\":\"用一句话介绍上海\",\"max_output_tokens\":32}")
printf '  HTTP %s\n' "$(echo "$r"|tail -1)"; echo "$r"|head -1|head -c 300

# ── 9. 批量 ──────────────────────────────────────────
hr "9. 批量 /v1/chat/completions/batch"
r=$(req POST /v1/chat/completions/batch "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"1+1=?\"}],\"max_tokens\":16}")
printf '  HTTP %s\n' "$(echo "$r"|tail -1)"; echo "$r"|head -1|head -c 200

# ── 10. SageMaker ────────────────────────────────────
hr "10. SageMaker /invocations"
r=$(req POST /invocations "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":16}")
printf '  HTTP %s\n' "$(echo "$r"|tail -1)"; echo "$r"|head -1|head -c 200

# ── 11. Render（需 chat template） ───────────────────
hr "11. Render / derender"
r=$(req POST /v1/chat/completions/render "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}]}")
printf '  HTTP %s\n' "$(echo "$r"|tail -1)"; echo "$r"|head -1|head -c 200

# ── 12. Scoring ──────────────────────────────────────
hr "12. /generative_scoring"
r=$(req POST /generative_scoring "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8}")
printf '  HTTP %s\n' "$(echo "$r"|tail -1)"; echo "$r"|head -1|head -c 150

# ── 13. Ping ─────────────────────────────────────────
hr "13. /ping"
for m in GET POST; do
  if [ "$m" = "GET" ]; then
    c=$(curl -s -o /dev/null -w '%{http_code}' "$BASE/ping")
  else
    c=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/ping")
  fi
  echo "  $m /ping → $c"
done

# ── 14. 生成接口（内部） ─────────────────────────────
hr "14. /inference/v1/generate"
r=$(req POST /inference/v1/generate "{\"model\":\"$MODEL\",\"prompt\":\"北京是\",\"max_tokens\":16}")
printf '  HTTP %s\n' "$(echo "$r"|tail -1)"; echo "$r"|head -1|head -c 200

# ── 15. Metrics（KV cache / 命中率） ─────────────────
hr "15. /metrics —— 与你研究的 KV cache 直接对应"
curl -s "$BASE/metrics" | grep -E \
  'vllm:(gpu_cache_usage_perc|cpu_cache_usage_perc|num_requests_(waiting|running)|prefix_cache_(queries|hits)|num_preemptions|kv_cache_usage)' \
  | head -20 | sed 's/^/  /'

hr "完成"
echo "提示: /docs 是 Swagger UI，浏览器打开 $BASE/docs 可交互调试"
