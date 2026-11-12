#!/bin/bash

# launch_external_lb_ascend.sh — 一键启动 vLLM Ascend External LB 模式 (DP+EP)
#
# 用法:
#   bash launch_external_lb_ascend.sh [OPTIONS]
#
# 示例:
#   bash launch_external_lb_ascend.sh
#   bash launch_external_lb_ascend.sh --dp 4 --devices 0,1,2,3
#   bash launch_external_lb_ascend.sh --model /path/to/model --dynamic-eplb true

set -uo pipefail

# ======================= 基础配置 =======================

DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"   # 使用的 NPU 设备编号，逗号分隔
DP_SIZE=24                                          # 全局数据并行大小（所有节点 rank 总数）
DP_LOCAL_SIZE=8                                     # 本节点启动的 DP rank 数（若小于 DP_SIZE，则本节点只负责一部分 rank）
DP_START_RANK=16                                    # 本节点第一个 DP rank 的全局编号（默认 0；双机按节点分片时需设置）
NODE_RANK=8                                         # 本节点编号（0 起）
DATA_PARALLEL_ADDRESS="80.5.9.139"                  # rank0 所在节点 IP（双机/多机必填）
DATA_PARALLEL_RPC_PORT=13345                        # 跨节点 DP RPC 固定端口（所有节点必须一致）
MODEL="/mnt/weight/DeepSeekV4-pro-0813-w4a8/"        # 模型路径
BASE_PORT=8211                                      # 基础端口，rank i 使用 port+i
GPU_MEM_UTIL=0.9                                    # NPU 显存利用率
MAX_MODEL_LEN=51200                                 # 最大序列长度
STARTUP_TIMEOUT=6000                                # 启动超时（秒）
TRUST_REMOTE_CODE=true                              # 是否信任远程代码
LOG_BASE="/vllm-workspace/vllm/logs"                # 日志根目录
EXTRA_ARGS=""                                       # 额外的 vllm serve 参数
DETACH=false                                        # Detach 模式：启动成功后退出，进程后台运行
CPU_DIST_TIMEOUT_SEC="120"                          # CPU 分布式组超时（秒）
QUANTIZATION="ascend"                               # 量化类型
MAX_NUM_BATCHED_TOKENS=32                           # 最大批处理 token 数
TP_SIZE=2
#COMPILATION_CONFIG='{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[4,8,12,16,20,24,28,32]}'
COMPILATION_CONFIG='{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[12, 24, 32]}'
KV_TRANSFER_CONFIG='{"kv_connector": "MooncakeHybridConnector", "kv_role": "kv_consumer", "kv_port": "36200", "kv_connector_extra_config": {"prefill": {"dp_size": 16, "tp_size": 2}, "decode": {"dp_size": 24, "tp_size": 2}}}'
#ADDITION_CONFIG='{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding": true,"multistream_overlap_shared_expert": true,"recompute_scheduler_enable":true}'
ADDITION_CONFIG='{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding": true,"multistream_overlap_shared_expert": true}'

# ======================= Ascend 特定配置 =======================

export VLLM_VERSION=0.34.0
export HCCL_BUFFSIZE=2400
export HCCL_EXEC_TIMEOUT=1600
export ASCEND_RT_VISIBLE_DEVICES=0
export HCCL_ASYNC_ERROR_HANDLING=0
export TASK_QUEUE_ENABLE=0
export VLLM_USE_V2_MODEL_RUNNER=1
export MULTI_STREAM_MEMORY_REUSE=2
export HCCL_IF_IP=80.5.9.143
export GLOO_SOCKET_IFNAME="enp194s0f0"
export TP_SOCKET_IFNAME="enp194s0f0"
export HCCL_SOCKET_IFNAME="enp194s0f0"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_OP_EXPANSION_MODE="AIV"

# ======================= EPLB 配置 =======================

EXPERT_HEAT_COLLECTION_INTERVAL=200                # 专家热度收集间隔
EPLB_ALGORITHM_EXECUTION_INTERVAL=50               # EPLB 算法执行间隔
NUM_REDUNDANT_EXPERTS=48                           # 冗余专家数量
ENABLE_EPLB="true"                                 # 是否启用动态 EPLB (你的配置是 false)
USE_ASYNC="true"

# ======================= 容错配置 =======================

ENABLE_FT=true                                     # 是否启用容错框架
ENGINE_RECOVERY_TIMEOUT_SEC=32000                  # 引擎恢复等待超时（毫秒）

# ======================= 帮助信息 =======================

usage() {
    cat <<EOF
用法: $(basename "$0") [OPTIONS]

一键启动 vLLM Ascend External LB 模式，每个 DP rank 独立进程。

基础选项:
  --devices DEVICES         NPU 设备编号，逗号分隔 (默认：$DEVICES)
  --dp N                    全局数据并行大小 (默认：$DP_SIZE)
  --model PATH              模型路径 (默认：$MODEL)
  --port N                  基础端口 (默认：$BASE_PORT)
  --gpu-mem F               显存利用率 (默认：$GPU_MEM_UTIL)
  --max-model-len N         最大序列长度 (默认：$MAX_MODEL_LEN)
  --timeout N               启动超时秒数 (默认：$STARTUP_TIMEOUT)
  --log-dir PATH            日志根目录 (默认：$LOG_BASE)
  --cpu-dist-timeout N      CPU 分布式组超时秒数 (默认：$CPU_DIST_TIMEOUT_SEC)
  --quantization TYPE       量化类型 (默认：$QUANTIZATION)
  --extra "ARGS"            传递给 vllm serve 的额外参数

双机/多机 (External LB DP 跨节点):
  --dp-local N              本节点启动的 DP rank 数，须 <= 全局 --dp (默认：等于 --dp)
  --dp-start RANK           本节点第一个 DP rank 的全局编号 (默认：0)
  --node-rank N             本节点编号，0 起 (默认：0)
  --dp-address IP           rank0 所在节点 IP，双机组网必填 (默认：空=单机)
  --dp-rpc-port PORT        跨节点 DP RPC 固定端口，所有节点一致 (默认：$DATA_PARALLEL_RPC_PORT)

EPLB 选项:
  --heat-interval N         专家热度收集间隔 (默认：$EXPERT_HEAT_COLLECTION_INTERVAL)
  --exec-interval N         算法执行间隔 (默认：$EPLB_ALGORITHM_EXECUTION_INTERVAL)
  --redundant-experts N     冗余专家数量 (默认：$NUM_REDUNDANT_EXPERTS)

容错选项:
  --ft                      启用容错框架
  --recovery-timeout N      引擎恢复超时毫秒数 (默认：$ENGINE_RECOVERY_TIMEOUT_SEC)
  --max-batched-tokens N    最大批处理 token 数 (默认：$MAX_NUM_BATCHED_TOKENS)
  --compilation-config JSON compilation config JSON (默认：cudagraph 优化)

其他:
  --detach                  启动成功后退出脚本，进程在后台继续运行
  -h, --help                显示此帮助信息
EOF
    exit 0
}

# ======================= 解析命令行参数 =======================

while [[ $# -gt 0 ]]; do
    case "$1" in
        --devices)
            DEVICES="$2"
            shift 2
            ;;
        --dp)
            DP_SIZE="$2"
            shift 2
            ;;
        --dp-local)
            DP_LOCAL_SIZE="$2"
            shift 2
            ;;
        --dp-start)
            DP_START_RANK="$2"
            shift 2
            ;;
        --node-rank)
            NODE_RANK="$2"
            shift 2
            ;;
        --dp-address)
            DATA_PARALLEL_ADDRESS="$2"
            shift 2
            ;;
        --dp-rpc-port)
            DATA_PARALLEL_RPC_PORT="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --port)
            BASE_PORT="$2"
            shift 2
            ;;
        --gpu-mem)
            GPU_MEM_UTIL="$2"
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --timeout)
            STARTUP_TIMEOUT="$2"
            shift 2
            ;;
        --log-dir)
            LOG_BASE="$2"
            shift 2
            ;;
        --cpu-dist-timeout)
            CPU_DIST_TIMEOUT_SEC="$2"
            shift 2
            ;;
        --quantization)
            QUANTIZATION="$2"
            shift 2
            ;;
        --heat-interval)
            EXPERT_HEAT_COLLECTION_INTERVAL="$2"
            shift 2
            ;;
        --exec-interval)
            EPLB_ALGORITHM_EXECUTION_INTERVAL="$2"
            shift 2
            ;;
        --redundant-experts)
            NUM_REDUNDANT_EXPERTS="$2"
            shift 2
            ;;
        --recovery-timeout)
            ENGINE_RECOVERY_TIMEOUT_SEC="$2"
            shift 2
            ;;
        --max-batched-tokens)
            MAX_NUM_BATCHED_TOKENS="$2"
            shift 2
            ;;
        --ft)
            ENABLE_FT=true
            shift
            ;;
        --extra)
            EXTRA_ARGS="$2"
            shift 2
            ;;
        --compilation-config)
            COMPILATION_CONFIG="$2"
            shift 2
            ;;
        --detach)
            DETACH=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "未知选项：$1"
            usage
            ;;
    esac
done

# ======================= 参数校验 =======================

IFS=',' read -ra DEVICE_LIST <<< "$DEVICES"
REQUIRED_DEVICES=$((DP_LOCAL_SIZE * TP_SIZE))
if [[ ${#DEVICE_LIST[@]} -lt $REQUIRED_DEVICES ]]; then
    echo "错误：本节点启动 --dp-local $DP_LOCAL_SIZE x --tp $TP_SIZE 需要至少 $REQUIRED_DEVICES 个 NPU 设备，但只指定了 ${#DEVICE_LIST[@]} 个 ($DEVICES)"
    exit 1
fi
if [[ "$DP_LOCAL_SIZE" -gt "$DP_SIZE" ]]; then
    echo "错误：--dp-local $DP_LOCAL_SIZE 不能大于全局 --dp $DP_SIZE"
    exit 1
fi
END_RANK=$((DP_START_RANK + DP_LOCAL_SIZE - 1))
if [[ "$END_RANK" -ge "$DP_SIZE" ]]; then
    echo "错误：--dp-start $DP_START_RANK + --dp-local $DP_LOCAL_SIZE 超出全局 DP ($DP_SIZE)"
    exit 1
fi

# ======================= 创建日志目录 =======================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${LOG_BASE}/node${NODE_RANK}*dp${DP_SIZE}*${QUANTIZATION}_${TIMESTAMP}"
mkdir -p "$LOG_DIR"
COMBINED_LOG="${LOG_DIR}/all_ranks.log"

# ======================= 打印启动配置 =======================

echo "============================================"
echo "  vLLM Ascend External LB 启动配置"
echo "============================================"
echo "  模型路径：       $MODEL"
echo "  全局 DP：        $DP_SIZE   (本节点启动 $DP_LOCAL_SIZE 个 rank: $DP_START_RANK ~ $END_RANK)"
echo "  本节点编号：     $NODE_RANK"
echo "  TP size：        $TP_SIZE"
echo "  NPU 设备列表：   ${DEVICE_LIST[*]:0:$REQUIRED_DEVICES}"
if [[ "$TP_SIZE" -gt 1 ]]; then
    for ((R=0; R<DP_LOCAL_SIZE; R++)); do
        RANK_DEVICES=""
        for ((T=0; T<TP_SIZE; T++)); do
            IDX=$((R * TP_SIZE + T))
            if [[ -n "$RANK_DEVICES" ]]; then
                RANK_DEVICES+=","
            fi
            RANK_DEVICES+="${DEVICE_LIST[$IDX]}"
        done
        echo "    Rank $((DP_START_RANK + R)) → $RANK_DEVICES"
    done
fi
if [[ -n "$DATA_PARALLEL_ADDRESS" ]]; then
    echo "  跨节点 DP：     地址=$DATA_PARALLEL_ADDRESS  RPC端口=$DATA_PARALLEL_RPC_PORT"
fi
echo "  量化类型：       $QUANTIZATION"
echo "  专家并行：       启用"
echo "    热度收集间隔：  $EXPERT_HEAT_COLLECTION_INTERVAL"
echo "    算法执行间隔：  $EPLB_ALGORITHM_EXECUTION_INTERVAL"
echo "    冗余专家数：    $NUM_REDUNDANT_EXPERTS"
echo "  容错框架：       $ENABLE_FT"
echo "  CPU 分布式超时：  ${CPU_DIST_TIMEOUT_SEC}s"
echo "  基础端口：       $BASE_PORT"
echo "  显存利用率：     $GPU_MEM_UTIL"
echo "  最大序列长度：   $MAX_MODEL_LEN"
echo "  最大批处理 token: $MAX_NUM_BATCHED_TOKENS"
echo "  启动超时：       ${STARTUP_TIMEOUT}s"
echo "  日志目录：       $LOG_DIR"
if [[ -n "$EXTRA_ARGS" ]]; then
    echo "  额外参数：       $EXTRA_ARGS"
fi
echo "  Detach 模式：    $DETACH"
echo "============================================"
echo ""
echo "  实时查看日志:"
echo "    单 rank: tail -f $LOG_DIR/rank_0.log"
echo "    汇总：    tail -f $COMBINED_LOG"

# ======================= 清理函数 =======================

PIDS=()
CLEANED=false

cleanup() {
    if [[ "$CLEANED" == "true" ]]; then return; fi
    CLEANED=true
    echo ""
    echo "正在清理所有 vLLM 进程..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    sleep 2
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "  进程 $pid 未响应 SIGTERM，强制终止..."
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    wait 2>/dev/null || true
    echo "所有进程已终止。"
    echo "日志保存在：$LOG_DIR"
}

trap cleanup INT TERM

# ======================= 构建 vLLM 启动参数 =======================

VLLM_ARGS=("$MODEL")
VLLM_ARGS+=("--tensor-parallel-size" "$TP_SIZE")
VLLM_ARGS+=("--data-parallel-size" "$DP_SIZE")
VLLM_ARGS+=("--trust-remote-code")
VLLM_ARGS+=("--gpu-memory-utilization" "$GPU_MEM_UTIL")
VLLM_ARGS+=("--max-model-len" "$MAX_MODEL_LEN")
VLLM_ARGS+=("--enable_expert_parallel")
VLLM_ARGS+=("--quantization" "$QUANTIZATION")
VLLM_ARGS+=("--max-num-batched-tokens" "$MAX_NUM_BATCHED_TOKENS")
VLLM_ARGS+=("--cpu-distributed-timeout-seconds" "$CPU_DIST_TIMEOUT_SEC")

VLLM_ARGS+=("--max-num-seqs" "4")
VLLM_ARGS+=("--tokenizer-mode" "deepseek_v4")
VLLM_ARGS+=("--tool-call-parser" "deepseek_v4")
VLLM_ARGS+=("--enable-auto-tool-choice")
VLLM_ARGS+=("--reasoning-parser" "deepseek_v4")
VLLM_ARGS+=("--no-enable-prefix-caching")
VLLM_ARGS+=("--no-disable-hybrid-kv-cache-manager")
VLLM_ARGS+=("--model-loader-extra-config={\"enable_multithread_load\": true, \"num_threads\": 128}")
VLLM_ARGS+=("--block-size" "32")
VLLM_ARGS+=("--additional-config" "$ADDITION_CONFIG")
VLLM_ARGS+=("--speculative-config={\"num_speculative_tokens\":5,\"method\":\"dspark\",\"enforce_eager\":true}")

if [[ -n "$COMPILATION_CONFIG" ]]; then
    VLLM_ARGS+=("--compilation-config" "$COMPILATION_CONFIG")
fi

if [[ -n "$KV_TRANSFER_CONFIG" ]]; then
    VLLM_ARGS+=("--kv-transfer-config" "$KV_TRANSFER_CONFIG")
fi

if [[ "$ENABLE_FT" == "true" ]]; then
    VLLM_ARGS+=("--enable-fault-tolerance")
    FT_CONFIG="{\"engine_recovery_timeout_sec\":$ENGINE_RECOVERY_TIMEOUT_SEC}"
    VLLM_ARGS+=("--fault-tolerance-config" "$FT_CONFIG")
fi

if [[ "$ENABLE_EPLB" == "true" ]]; then
    VLLM_ARGS+=("--enable-eplb")
    EPLB_CONFIG="{\"num_redundant_experts\":$NUM_REDUNDANT_EXPERTS, \"use_async\": $USE_ASYNC}"
    VLLM_ARGS+=("--eplb-config" "$EPLB_CONFIG")
fi

#VLLM_ARGS+=("--eplb-config.step_interval" "3000")
#VLLM_ARGS+=("--eplb-config.log_balancedness" "true")
#VLLM_ARGS+=("--eplb-config.log_balancedness_interval" "10")

# ======================= 启动各 Rank 进程 =======================

echo "正在启动 $DP_SIZE 个 rank..."
echo ""

# 打印完整启动命令（便于调试）

echo "============================================"
echo "  启动命令预览"
echo "============================================"
echo "环境变量:"
echo "  HCCL_BUFFSIZE=$HCCL_BUFFSIZE"
echo "  HCCL_EXEC_TIMEOUT=$HCCL_EXEC_TIMEOUT"
echo ""
echo "vllm serve 命令模板 (每个 rank):"
echo "  cd /vllm-workspace/vllm && \\"
echo "  ASCEND_RT_VISIBLE_DEVICES=<NPU_ID> \\"
echo "  vllm serve ${VLLM_ARGS[*]} \\"
echo "      --data-parallel-rank <GLOBAL_RANK> \\"
echo "      --port <PORT> \\"
if [[ -n "$DATA_PARALLEL_ADDRESS" ]]; then
    echo "      --data-parallel-address $DATA_PARALLEL_ADDRESS \\"
    echo "      --data-parallel-rpc-port $DATA_PARALLEL_RPC_PORT \\"
fi
echo "      $EXTRA_ARGS"
echo "============================================"
echo ""

# 设置环境变量（在循环外设置一次即可）

export HCCL_EXEC_TIMEOUT=1

for ((L=0; L<DP_LOCAL_SIZE; L++)); do
    GLOBAL_RANK=$((DP_START_RANK + L))
    R=$L

    # 当 TP > 1 时，每个 rank 需要分配多个设备

    # 计算该 rank 需要的设备数量

    if [[ "$TP_SIZE" -gt 1 ]]; then

        # 收集该 rank 需要的 TP_SIZE 个设备

        DEVICE=""
        for ((T=0; T<TP_SIZE; T++)); do
            IDX=$((R * TP_SIZE + T))
            if [[ $IDX -lt ${#DEVICE_LIST[@]} ]]; then
                if [[ -n "$DEVICE" ]]; then
                    DEVICE+=","
                fi
                DEVICE+="${DEVICE_LIST[$IDX]}"
            fi
        done
    else
        DEVICE="${DEVICE_LIST[$R]}"
    fi
    PORT=$((BASE_PORT + R))
    LOG_FILE="$LOG_DIR/rank_${GLOBAL_RANK}.log"
    DATA_PARALLEL_RANK=$GLOBAL_RANK

    # 打印当前 rank 的完整环境变量（同时写入日志文件）

    {
        echo "============================================"
        echo "  [Rank $GLOBAL_RANK] 环境变量"
        echo "============================================"
        echo "    ASCEND_RT_VISIBLE_DEVICES=$DEVICE"
        echo "    HCCL_BUFFSIZE=$HCCL_BUFFSIZE"
        echo "    HCCL_EXEC_TIMEOUT=$HCCL_EXEC_TIMEOUT"
        echo ""
        echo "  启动命令:"
        echo -n "    ASCEND_RT_VISIBLE_DEVICES=$DEVICE vllm serve ${VLLM_ARGS[*]} --data-parallel-rank $GLOBAL_RANK --port $PORT"
        if [[ -n "$DATA_PARALLEL_ADDRESS" ]]; then
            echo -n " --data-parallel-address $DATA_PARALLEL_ADDRESS --data-parallel-rpc-port $DATA_PARALLEL_RPC_PORT"
        fi
        echo " $EXTRA_ARGS"
        echo "============================================"
    } | tee -a "$LOG_FILE"
    echo ""

    DP_MULTI_ARGS=()
    if [[ -n "$DATA_PARALLEL_ADDRESS" ]]; then
        DP_MULTI_ARGS+=("--data-parallel-address" "$DATA_PARALLEL_ADDRESS")
        DP_MULTI_ARGS+=("--data-parallel-rpc-port" "$DATA_PARALLEL_RPC_PORT")
    fi

    # 使用 env 命令为每个 rank 独立设置环境变量（避免相互影响）

    if [[ "$DETACH" == "true" ]]; then
        cd /vllm-workspace/vllm &&
            env ASCEND_RT_VISIBLE_DEVICES="$DEVICE" \
                HCCL_BUFFSIZE=2400 \
                HCCL_EXEC_TIMEOUT=1 \
                vllm serve \
                "${VLLM_ARGS[@]}" \
                --data-parallel-rank "$DATA_PARALLEL_RANK" \
                --port "$PORT" \
                "${DP_MULTI_ARGS[@]}" \
                $EXTRA_ARGS \
                > "$LOG_FILE" 2>&1 &
    else
        cd /vllm-workspace/vllm &&
            env ASCEND_RT_VISIBLE_DEVICES="$DEVICE" \
                HCCL_BUFFSIZE=2400 \
                HCCL_EXEC_TIMEOUT=1 \
                vllm serve \
                "${VLLM_ARGS[@]}" \
                --data-parallel-rank "$DATA_PARALLEL_RANK" \
                --port "$PORT" \
                "${DP_MULTI_ARGS[@]}" \
                $EXTRA_ARGS \
                2>&1 | tee "$LOG_FILE" | sed -u "s/^/[R$GLOBAL_RANK] /" >> "$COMBINED_LOG" &
    fi

    PIDS+=($!)
    echo "  => PID=${PIDS[$R]}  全局Rank=$GLOBAL_RANK  端口=$PORT  日志=$LOG_FILE"
    echo ""

done

echo ""
echo "等待服务启动 (超时：${STARTUP_TIMEOUT}s)..."
echo ""

# ======================= 监控启动状态 =======================

READY=()
for ((L=0; L<DP_LOCAL_SIZE; L++)); do
    READY+=("false")
done

LAST_PROGRESS=0
START=$(date +%s)

while true; do
    ELAPSED=$(( $(date +%s) - START ))

    # 超时检查

    if [[ $ELAPSED -ge $STARTUP_TIMEOUT ]]; then
        PENDING=0
        for ((L=0; L<DP_LOCAL_SIZE; L++)); do
            if [[ "${READY[$L]}" != "true" && "${READY[$L]}" != "exited" ]]; then
                PENDING=$((PENDING + 1))
            fi
        done
        if [[ $PENDING -gt 0 ]]; then
            echo ""
            echo "启动超时 (${STARTUP_TIMEOUT}s)，以下 rank 未就绪:"
            for ((L=0; L<DP_LOCAL_SIZE; L++)); do
                if [[ "${READY[$L]}" != "true" && "${READY[$L]}" != "exited" ]]; then
                    GR=$((DP_START_RANK + L))
                    echo "=== Rank $GR (最后 40 行) ==="
                    tail -40 "$LOG_DIR/rank_${GR}.log" 2>/dev/null || true
                    echo ""
                fi
            done
        fi
        break
    fi

    # 逐个检查 rank 状态

    READY_COUNT=0
    EXITED_COUNT=0
    for ((L=0; L<DP_LOCAL_SIZE; L++)); do
        PID="${PIDS[$L]}"
        GR=$((DP_START_RANK + L))

        if [[ "${READY[$L]}" == "true" ]]; then
            READY_COUNT=$((READY_COUNT + 1))
            continue
        fi
        if [[ "${READY[$L]}" == "exited" ]]; then
            EXITED_COUNT=$((EXITED_COUNT + 1))
            continue
        fi

        # 检查进程是否崩溃

        if ! kill -0 "$PID" 2>/dev/null; then
            echo "注意：Rank $GR (PID $PID) 在启动 ${ELAPSED}s 后退出"
            echo "=== Rank $GR 日志 (最后 40 行) ==="
            tail -40 "$LOG_DIR/rank_${GR}.log" 2>/dev/null || true
            echo ""
            READY[$L]="exited"
            EXITED_COUNT=$((EXITED_COUNT + 1))
            continue
        fi

        # 检查是否已就绪

        if grep -q "Application startup complete" "$LOG_DIR/rank_${GR}.log" 2>/dev/null; then
            READY[$L]="true"
            READY_COUNT=$((READY_COUNT + 1))
            PORT=$((BASE_PORT + L))
            echo "  [${ELAPSED}s] Rank $GR 已就绪 (端口 $PORT)"
        fi
    done

    # 全部 rank 都已退出

    if [[ $EXITED_COUNT -eq $DP_LOCAL_SIZE ]]; then
        echo ""
        echo "所有本节点 rank 均在启动阶段退出，请检查日志：$LOG_DIR"
        exit 1
    fi

    # 所有存活 rank 就绪

    if [[ $((READY_COUNT + EXITED_COUNT)) -eq $DP_LOCAL_SIZE && $READY_COUNT -gt 0 ]]; then
        echo ""
        echo "============================================"
        if [[ $EXITED_COUNT -gt 0 ]]; then
            echo "  $READY_COUNT/$DP_LOCAL_SIZE 个本节点 rank 已就绪，$EXITED_COUNT 个已退出 (耗时 ${ELAPSED}s)"
        else
            echo "  本节点全部 $DP_LOCAL_SIZE 个 rank 已就绪 (耗时 ${ELAPSED}s)"
        fi
        echo "============================================"
        for ((L=0; L<DP_LOCAL_SIZE; L++)); do
            GR=$((DP_START_RANK + L))
            PORT=$((BASE_PORT + L))
            if [[ "${READY[$L]}" == "true" ]]; then
                echo "  Rank $GR → http://0.0.0.0:$PORT/v1"
            else
                echo "  Rank $GR → (已退出)"
            fi
        done
        echo ""
        echo "  日志目录：$LOG_DIR"
        echo "  汇总日志：$COMBINED_LOG"
        if [[ -n "$DATA_PARALLEL_ADDRESS" ]]; then
            echo ""
            echo "  [双机/多机] 需在所有节点启动完成后，DP 组才会就绪"
        fi
        if [[ "$DETACH" == "true" ]]; then
            echo ""
            echo "  [Detach 模式] 进程在后台继续运行"
            echo "  停止服务：pkill -f 'vllm serve'"
        else
            echo ""
            echo "  按 Ctrl+C 停止所有服务"
        fi
        echo "============================================"
        break
    fi

    # 定期输出进度

    if [[ $((ELAPSED - LAST_PROGRESS)) -ge 15 ]]; then
        LAST_PROGRESS=$ELAPSED
        echo "  [${ELAPSED}s] $READY_COUNT/$DP_LOCAL_SIZE 本节点已就绪，$EXITED_COUNT 已退出..."
        for ((L=0; L<DP_LOCAL_SIZE; L++)); do
            if [[ "${READY[$L]}" != "true" && "${READY[$L]}" != "exited" ]]; then
                GR=$((DP_START_RANK + L))
                LAST_LINE=$(tail -1 "$LOG_DIR/rank_${GR}.log" 2>/dev/null | head -c 160)
                if [[ -n "$LAST_LINE" ]]; then
                    echo "         rank $GR: $LAST_LINE"
                fi
                break
            fi
        done
    fi

    sleep 3

done

# ======================= Detach 模式：启动成功后退出 =======================

if [[ "$DETACH" == "true" ]]; then
    printf '%s\n' "${PIDS[@]}" > "$LOG_DIR/pids.txt"
    trap - EXIT
    exit 0
fi

# ======================= 服务运行中 — 等待退出信号 =======================

EXITED=()
for ((L=0; L<DP_LOCAL_SIZE; L++)); do
    EXITED+=("false")
done

while true; do
    ALIVE_COUNT=0
    for ((L=0; L<DP_LOCAL_SIZE; L++)); do
        if [[ "${EXITED[$L]}" == "true" ]]; then
            continue
        fi
        if ! kill -0 "${PIDS[$L]}" 2>/dev/null; then
            GR=$((DP_START_RANK + L))
            EXITED[$L]="true"
            echo ""
            echo "注意：Rank $GR (PID ${PIDS[$L]}) 已退出"
            echo "=== Rank $GR 日志 (最后 40 行) ==="
            tail -40 "$LOG_DIR/rank_${GR}.log" 2>/dev/null || true
            echo ""
        else
            ALIVE_COUNT=$((ALIVE_COUNT + 1))
        fi
    done

    if [[ $ALIVE_COUNT -eq 0 ]]; then
        echo "本节点所有 rank 均已退出。"
        echo "日志保存在：$LOG_DIR"
        break
    fi

    sleep 5

done
