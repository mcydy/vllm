#!/usr/bin/env bash
# chmod +x vllm_lab.sh

#  看所有场景
# ./vllm_lab.sh list

#  先跑这两个建立基线
# ./vllm_lab.sh baseline
# ./vllm_lab.sh eager          # 关编译+graph，对比出"编译值多少"

# ./vllm_lab.sh cg-piecewise   # padding 浪费实测
# ./vllm_lab.sh prefix-off     # 前缀缓存收益
# ./vllm_lab.sh moe            # MoE 路径
# ./vllm_lab.sh spec-ngram     # 推测解码（无需 draft 模型）
# =============================================================================
# vLLM 功能域逐项实测脚本  (针对 RTX 3090 裁剪)
# 用法: ./vllm_lab.sh <场景名>      例: ./vllm_lab.sh baseline
#       ./vllm_lab.sh suite-compile  串行跑编译对比组
#       ./vllm_lab.sh list           列出所有场景
# =============================================================================
set -uo pipefail

# ---------- 环境配置 ----------
DENSE_MODEL=${DENSE_MODEL:-/home/jeff/work/demo/Qwen3-4B-Instruct-2507}
MOE_MODEL=${MOE_MODEL:-/home/jeff/work/demo/OLMoE-1B-7B-0924}
PORT=${PORT:-8000}
OUTDIR=${OUTDIR:-/tmp/vllm_lab}
mkdir -p "$OUTDIR"

IN_LEN=${IN_LEN:-1024}
OUT_LEN=${OUT_LEN:-128}
NUM_PROMPTS=${NUM_PROMPTS:-50}
CONCURRENCY=${CONCURRENCY:-8}

GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)

# ---------- 自动检测版本支持哪些观测参数 ----------
HELP_TXT=$(vllm serve --help 2>/dev/null || echo "")
has_opt() { echo "$HELP_TXT" | grep -q -- "$1" && echo 1 || echo 0; }

OBS_FLAGS=()
[[ "$(has_opt --enable-mfu-metrics)" == "1" ]] && OBS_FLAGS+=(--enable-mfu-metrics)
[[ "$(has_opt --kv-cache-metrics)"   == "1" ]] && OBS_FLAGS+=(--kv-cache-metrics)
[[ "$(has_opt --cudagraph-metrics)"  == "1" ]] && OBS_FLAGS+=(--cudagraph-metrics)

log()  { echo -e "\n\033[1;36m[$*]\033[0m"; }
warn() { echo -e "\033[1;33m[!] $*\033[0m"; }

# ---------- 等待服务 ready ----------
wait_ready() {
  echo -n "  等待服务启动"
  for i in $(seq 1 180); do
    if curl -s -m 2 "http://127.0.0.1:${PORT}/health" -o /dev/null 2>/dev/null; then
      echo " OK (${i}s)"; return 0
    fi
    sleep 1; echo -n "."
  done
  echo " 超时!"; return 1
}

# ---------- 抓取 Prometheus 指标 ----------
grab_metrics() {
  local tag=$1
  curl -s "http://127.0.0.1:${PORT}/metrics" > "${OUTDIR}/${tag}.metrics.txt" 2>/dev/null
  echo "  --- 关键指标 ---"
  grep -iE "prefix_cache_hit|cudagraph|kv_cache_usage|estimated_flops|estimated_read_bytes|estimated_write_bytes" \
      "${OUTDIR}/${tag}.metrics.txt" 2>/dev/null | grep -v "^#" | head -20
}

# ---------- 跑 bench ----------
run_bench() {
  local tag=$1
  log "bench: ${tag}"
  vllm bench serve \
      --backend vllm --base-url "http://127.0.0.1:${PORT}" \
      --model "$CUR_MODEL" --endpoint /v1/completions \
      --dataset-name random \
      --random-input-len "$IN_LEN" --random-output-len "$OUT_LEN" \
      --num-prompts "$NUM_PROMPTS" --max-concurrency "$CONCURRENCY" \
      2>&1 | tee "${OUTDIR}/${tag}.bench.txt" | tail -25
}

# ---------- 通用: 启动 -> bench -> 抓指标 -> 停 ----------
run_case() {
  local tag=$1; shift
  local extra=("$@")
  CUR_MODEL=${DENSE_MODEL}
  log "===== 场景: ${tag} ====="
  echo "  参数: ${extra[*]:-(默认)}"

  pkill -f "vllm serve" >/dev/null 2>&1; sleep 2

  # 打印完整启动命令
  printf '  [CMD] '; printf '%q ' vllm serve "$CUR_MODEL" \
      --host 127.0.0.1 --port "$PORT" \
      --enable-mfu-metrics   --enable-logging-iteration-details \
      --max-model-len 8192 --gpu-memory-utilization 0.85 \
      "${OBS_FLAGS[@]}" "${extra[@]}"; echo
  printf '%q ' vllm serve "$CUR_MODEL" \
      --host 127.0.0.1 --port "$PORT" \
      --enable-mfu-metrics   --enable-logging-iteration-details \
      --max-model-len 8192 --gpu-memory-utilization 0.85 \
      "${OBS_FLAGS[@]}" "${extra[@]}" > "${OUTDIR}/${tag}.cmd.txt"
  echo

  vllm serve "$CUR_MODEL" --host 127.0.0.1 --port "$PORT" \
      --enable-mfu-metrics   --enable-logging-iteration-details \
      --max-model-len 8192 --gpu-memory-utilization 0.85 \
      "${OBS_FLAGS[@]}" "${extra[@]}" \
      > "${OUTDIR}/${tag}.server.log" 2>&1 &
  local pid=$!

  if ! wait_ready; then
    warn "服务启动失败, 看日志: ${OUTDIR}/${tag}.server.log"
    tail -30 "${OUTDIR}/${tag}.server.log"; kill $pid 2>/dev/null; return 1
  fi

  run_bench "$tag"
  grab_metrics "$tag"

  kill $pid 2>/dev/null; wait $pid 2>/dev/null
  echo "  日志: ${OUTDIR}/${tag}.server.log"
}

# =============================================================================
# 场景定义
# =============================================================================
case "${1:-list}" in

  # ---------- A. 编译与 CUDA graph ----------
  baseline)   run_case baseline ;;
  eager)      run_case eager --enforce-eager ;;
  compile0)   run_case compile0 --compilation-config '{"level":0}' ;;
  compile1)   run_case compile1 --compilation-config '{"level":1}' ;;
  compile2)   run_case compile2 --compilation-config '{"level":2}' ;;
  compile3)   run_case compile3 --compilation-config '{"level":3}' ;;
  cg-full)      run_case cg-full      --compilation-config '{"cudagraph_mode":"FULL"}' ;;
  cg-piecewise) run_case cg-piecewise --compilation-config '{"cudagraph_mode":"PIECEWISE"}' ;;

  suite-compile)
    for c in baseline eager compile0 compile1 compile2 compile3 cg-full cg-piecewise; do
      ./$0 $c || warn "$c 失败, 继续"
    done
    log "编译对比组完成, 结果对比:"
    for c in baseline eager compile0 compile1 compile2 compile3 cg-full cg-piecewise; do
      f="${OUTDIR}/${c}.bench.txt"
      [[ -f "$f" ]] && echo "  $c: $(grep -iE 'Mean TTFT|Mean TPOT|Output token throughput' "$f" | tr '\n' ' ')"
    done
    ;;

  # ---------- B. MoE ----------
  moe)
    CUR_MODEL=$MOE_MODEL
    log "===== 场景: moe ($MOE_MODEL) ====="
    pkill -f "vllm serve" >/dev/null 2>&1; sleep 2
    vllm serve "$MOE_MODEL" --host 127.0.0.1 --port "$PORT" \
        --max-model-len 4096 --gpu-memory-utilization 0.85 \
        "${OBS_FLAGS[@]}" > "${OUTDIR}/moe.server.log" 2>&1 &
    local_pid=$!
    wait_ready || { tail -30 "${OUTDIR}/moe.server.log"; kill $local_pid; exit 1; }
    run_bench moe
    grab_metrics moe
    echo "  --- 确认 MoE kernel 是否启用 ---"
    grep -iE "fused_moe|moe_align|Using.*moe|expert" "${OUTDIR}/moe.server.log" | head -10
    kill $local_pid 2>/dev/null
    ;;

  # ---------- E. Attention 后端 ----------
  attn-triton)
    VLLM_ATTENTION_BACKEND=TRITON_ATTN run_case attn-triton ;;
  attn-flex)
    VLLM_ATTENTION_BACKEND=FLEX_ATTENTION run_case attn-flex ;;

  suite-attn)
    ./$0 baseline; ./$0 attn-triton; ./$0 attn-flex
    for c in baseline attn-triton attn-flex; do
      f="${OUTDIR}/${c}.bench.txt"
      [[ -f "$f" ]] && echo "  $c: $(grep -iE 'Mean TTFT|Mean TPOT' "$f" | tr '\n' ' ')"
    done
    ;;

  # ---------- F/G. 前缀缓存 / 调度 / 推测解码 ----------
  prefix-off)  run_case prefix-off --no-enable-prefix-caching ;;
  async-sched) run_case async-sched --async-scheduling ;;
  maxseqs-4)   run_case maxseqs-4  --max-num-seqs 4 ;;
  maxseqs-256) run_case maxseqs-256 --max-num-seqs 256 ;;

  spec-ngram)
    run_case spec-ngram --speculative-config \
      '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4}' ;;

  # ---------- 结构化输出 ----------
  structured)
    run_case structured
    log "结构化输出验证"
    curl -s "http://127.0.0.1:${PORT}/v1/completions" \
      -H "Content-Type: application/json" \
      -d '{"model":"'"$DENSE_MODEL"'","prompt":"Give me a JSON object with name and age fields.",
           "max_tokens":64,"temperature":0,
           "response_format":{"type":"json_schema","json_schema":{"name":"p","schema":{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"]}}}}' \
      | head -c 600
    echo
    ;;

  # ---------- G. 只抓指标, 不压测 ----------
  kv-metrics)
    CUR_MODEL=$DENSE_MODEL
    log "===== 场景: kv-metrics (只启服务抓指标) ====="
    pkill -f "vllm serve" >/dev/null 2>&1; sleep 2
    vllm serve "$DENSE_MODEL" --host 127.0.0.1 --port "$PORT" \
        --max-model-len 8192 --gpu-memory-utilization 0.85 \
        "${OBS_FLAGS[@]}" > "${OUTDIR}/kv-metrics.server.log" 2>&1 &
    mp=$!
    wait_ready || { tail -30 "${OUTDIR}/kv-metrics.server.log"; kill $mp; exit 1; }
    echo "  发 3 次相同 prompt, 观察 prefix cache 命中率变化..."
    P=$(python3 -c "print('The history of artificial intelligence began in antiquity, with myths and stories. '*40)")
    for i in 1 2 3; do
      curl -s "http://127.0.0.1:${PORT}/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$DENSE_MODEL\",\"prompt\":\"$P\",\"max_tokens\":16,\"temperature\":0}" > /dev/null
      echo "    第 $i 次后命中率:"
      curl -s "http://127.0.0.1:${PORT}/metrics" | grep -i prefix_cache_hit | grep -v '^#' | head -3
    done
    grab_metrics kv-metrics
    kill $mp 2>/dev/null
    ;;

  # ---------- D. PD 分离 (单机两实例 + 共享存储) ----------
  pd-demo)
    log "===== 场景: PD 分离 (SharedStorageConnector) ====="
    if [[ "$GPU_COUNT" -lt 1 ]]; then warn "需要至少 1 张 GPU"; exit 1; fi
    mkdir -p /tmp/vllm_pd_store
    echo "  [提示] 单机 PD 分离主要验证链路连通, 性能无意义(两实例抢同一张卡)"

    echo "  启动 Producer (prefill)..."
    VLLM_NIXL_SIDE_CHANNEL_PORT=5555 \
    vllm serve "$DENSE_MODEL" --host 127.0.0.1 --port 8100 \
        --max-model-len 8192 --gpu-memory-utilization 0.40 \
        --kv-transfer-config \
        '{"kv_connector":"SharedStorageConnector","kv_role":"kv_producer","kv_connector_extra_config":{"shared_storage_path":"/tmp/vllm_pd_store"}}' \
        > "${OUTDIR}/pd-producer.log" 2>&1 &
    pp=$!
    sleep 5

    echo "  启动 Consumer (decode)..."
    vllm serve "$DENSE_MODEL" --host 127.0.0.1 --port 8200 \
        --max-model-len 8192 --gpu-memory-utilization 0.40 \
        --kv-transfer-config \
        '{"kv_connector":"SharedStorageConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"shared_storage_path":"/tmp/vllm_pd_store"}}' \
        > "${OUTDIR}/pd-consumer.log" 2>&1 &
    cp=$!
    sleep 5

    for p in 8100 8200; do
      echo -n "  等待 $p ready"; for i in $(seq 1 90); do
        curl -s -m 2 "http://127.0.0.1:$p/health" -o /dev/null 2>/dev/null && { echo " OK"; break; }
        sleep 1; echo -n "."
      done
    done

    echo "  注意: 此模式需要外部 router 把 prefill 请求发 8100、decode 发 8200"
    echo "  这里仅验证两实例都能起来 + KV 目录有文件写入"
    sleep 3
    ls -la /tmp/vllm_pd_store | head -10
    echo "  producer 日志尾部:"; tail -5 "${OUTDIR}/pd-producer.log"
    echo "  consumer 日志尾部:"; tail -5 "${OUTDIR}/pd-consumer.log"
    read -r -p "  按回车停止两实例..." _
    kill $pp $cp 2>/dev/null
    ;;

  # ---------- 3090 不支持的: 给出明确提示 ----------
  fp8)
    log "FP8 量化 —— 3090 (SM86) 不支持"
    cat <<'EOF'
  ❌ FP8 Tensor Core 需要 SM89 (Ada/L40S/4090) 或 SM90 (H100) 以上
     3090 是 SM86 (Ampere), vLLM 会报:
       "FP8 is not supported on this device"
  ✅ 替代方案: 在 3090 上测 AWQ / GPTQ (INT4), 但需要准备量化权重:
       # 下载现成 AWQ 权重, 或自己量化
       vllm serve <awq-model> --quantization awq_marlin
  ⚠️ marlin kernel 也需 SM80+, 3090 可用
EOF
    ;;

  tp2)
    log "TP=2 —— 需要 2 张 GPU"
    if [[ "$GPU_COUNT" -lt 2 ]]; then
      warn "检测到 ${GPU_COUNT} 张 GPU, TP=2 无法运行"
    else
      run_case tp2 --tensor-parallel-size 2
    fi
    ;;

  # ---------- 全量 ----------
  suite-all)
    for c in baseline eager compile3 cg-piecewise moe attn-triton prefix-off async-sched spec-ngram; do
      ./$0 "$c" || warn "$c 失败, 继续下一个"
    done
    log "全部完成, 汇总:"
    for c in baseline eager compile3 cg-piecewise moe attn-triton prefix-off async-sched spec-ngram; do
      f="${OUTDIR}/${c}.bench.txt"
      [[ -f "$f" ]] && echo "  $c: $(grep -iE 'Mean TTFT|Mean TPOT|Output token throughput' "$f" | tr '\n' ' ')"
    done
    ;;

  list|*)
    cat <<EOF
可用场景:

  【A. 编译 / CUDA graph】—— 最能体现 vLLM 复杂度
    baseline        默认配置(基准线)
    eager           --enforce-eager, 关编译+graph, 最慢基线
    compile0/1/2/3  --compilation-config level=N
    cg-full         cudagraph_mode=FULL
    cg-piecewise    cudagraph_mode=PIECEWISE  ← 看 padding 浪费
    suite-compile   串行跑上面全部并汇总

  【B. MoE】
    moe             切到 OLMoE-1B-7B, 验证 fused_moe 路径

  【E. Attention 后端】
    attn-triton     TRITON_ATTN
    attn-flex       FLEX_ATTENTION
    suite-attn      对比组

  【F/G. 调度 / 缓存 / 推测解码】
    prefix-off      关闭前缀缓存, 对比命中率与 TTFT
    async-scheduling 异步调度(对应 DeepInfer Asynchronous Scheduling)
    maxseqs-4/256   改变 batch 上限
    spec-ngram      ngram 推测解码(无需 draft 模型)
    structured      JSON schema 结构化输出

  【G. 指标观测】
    kv-metrics      只启服务, 连发相同 prompt 看命中率爬升

  【D. PD 分离】
    pd-demo         SharedStorageConnector 单机两实例

  【3090 测不了的】
    fp8             打印为什么不支持 + 替代方案
    tp2             检测 GPU 数量, 不足则提示

  【全量】
    suite-all       跑所有可跑场景并汇总

环境变量:
  DENSE_MODEL  默认 $DENSE_MODEL
  MOE_MODEL    默认 $MOE_MODEL
  PORT         默认 8000
  IN_LEN/OUT_LEN/NUM_PROMPTS/CONCURRENCY  压测参数

结果输出目录: $OUTDIR
EOF
    ;;
esac