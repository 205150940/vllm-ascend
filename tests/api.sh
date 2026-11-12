#!/bin/bash

# api.sh — 一键向所有 DP rank 发送 API 请求
#
# 用法:
#   bash api.sh                          # 向所有 rank 发推理请求（默认 prompt）
#   bash api.sh chat "你好"              # 自定义 prompt
#   bash api.sh chat "1+1=?" --dp 2      # 只发 2 个 rank
#   bash api.sh status                   # 查询各 rank FT 状态
#   bash api.sh pause --round 1          # 向所有 rank 发 pause（暂停，仅 HEALTHY 引擎受理）
#   bash api.sh retry --round 1          # 向所有 rank 发 retry
#   bash api.sh health                   # 检查各 rank /health 端点
#   bash api.sh models                   # 查询各 rank 加载的模型
#   bash api.sh scale_down 1,3 --round 2 # 缩容: 移除 rank 1,3, 向所有 rank(含被移除的)发送 scale_down
#   bash api.sh scale_down 0 --round 3 --dp-store-port 29501 --dp-master-ip 127.0.0.1  # rank 0 被移除时后两者必填
#
# 多机: pause/retry/scale_down 必须显式给 --round N，且每个节点用同一个 N 并同时执行
# （详见 --round 的帮助说明）。

set -uo pipefail

# ======================= 默认配置 =======================

BASE_PORT=8211
DP_SIZE=8
HOST="0.0.0.0"
PROMPT="请用一句话介绍你自己。"
MAX_TOKENS=128
TIMEOUT=30
RETRY_TIMEOUT=120
REMOVED_RANKS=""
DP_STORE_PORT=""
DP_MASTER_IP=""
ROUND="${FT_ROUND:-}"

# --round N 或环境变量 FT_ROUND。pause/retry/scale_down 必填。

REQUEST_ID=""

# --request-id 显式指定完整 ID, 优先于 --round, 仅排障用。
# 为什么不再自动生成: 一轮 recovery 的 TCPStore rendezvous 键
# (ft_round_member_ / ft_recovery_store_port_ / ft_engine_dp_) 只按 request_id
# 命名且从不清理。同一轮所有 rank 必须同一个 ID(不同 ID = 两个互不可见的轮次,
# 必然点名超时); 跨轮绝不能复用(残留键会让下一轮点名"假通过", 然后卡在 gloo 重建上)。
# 自动时间戳在跨节点场景下必然产生不一致的 ID, 所以改为强制显式 --round。

# ======================= 帮助信息 =======================

usage() {
    cat <<EOF
用法: $(basename "$0") [COMMAND] [OPTIONS]

向所有 DP rank 并发发送 API 请求。

命令:
  chat [PROMPT]       发送推理请求 (默认)
  status              查询 FT 状态 (GET /v1/fault_tolerance/status)
  pause               发送 pause 指令 (POST /v1/fault_tolerance/apply)   [需 --round]
  retry               发送 retry 指令 (POST /v1/fault_tolerance/apply)   [需 --round]
  health              检查 /health 端点
  models              查询已加载模型 (/v1/models)
  scale_down RANKS    缩容: 移除指定 rank (逗号分隔, 如 1,3), 向所有 rank(含被移除的)发送 scale_down  [需 --round]

选项:
  --dp N              DP rank 数量 (默认: $DP_SIZE)
  --port N            基础端口 (默认: $BASE_PORT)
  --host H            服务器地址 (默认: $HOST)
  --max-tokens N      最大生成 token 数 (默认: $MAX_TOKENS)
  --timeout N         请求超时秒数 (默认: $TIMEOUT)
  --retry-timeout N   retry 超时秒数 (默认: $RETRY_TIMEOUT)
  --dp-store-port N   缩容时新 dp_store 端口 (rank 0 被移除时必填)
  --dp-master-ip IP   缩容时新 master IP (rank 0 被移除时必填)
  --round N           FT 轮次号 (pause/retry/scale_down 必填, 正整数, 也可用环境变量 FT_ROUND)。
                      实际 request_id = ft-<命令>-round-<N>, 因此:
                        - 同一条命令重发(同 N) -> 同 ID, 满足前端 scale_down 断点续跑的要求
                        - pause/retry/scale_down 之间 -> 自动分命名空间, 不会互相污染残留键
                      跨节点: 同一轮每个节点必须传同一个 N 并同时执行(点名窗口 10s);
                      每个新轮次换新的 N, 复用旧 N 会读到上一轮残留的 rendezvous 键。
  --request-id ID     直接指定完整 request_id, 覆盖 --round (仅排障用, 需自行保证跨节点一致)
  --seq / --serial    串行发送（默认并发）
  -h, --help          显示帮助
EOF
    exit 0
}

# ======================= 解析参数 =======================

COMMAND="chat"
SERIAL=false
POSITIONAL=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        chat|status|pause|retry|health|models|scale_down)
            COMMAND="$1"
            shift
            ;;
        --dp)
            DP_SIZE="$2"
            shift 2
            ;;
        --port)
            BASE_PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --retry-timeout)
            RETRY_TIMEOUT="$2"
            shift 2
            ;;
        --dp-store-port)
            DP_STORE_PORT="$2"
            shift 2
            ;;
        --dp-master-ip)
            DP_MASTER_IP="$2"
            shift 2
            ;;
        --round)
            ROUND="$2"
            shift 2
            ;;
        --request-id)
            REQUEST_ID="$2"
            shift 2
            ;;
        --seq|--serial)
            SERIAL=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        --*)
            echo "未知选项: $1"
            usage
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

if [[ ${#POSITIONAL[@]} -gt 0 ]]; then
    if [[ "$COMMAND" == "scale_down" ]]; then
        REMOVED_RANKS="${POSITIONAL[0]}"
    else
        PROMPT="${POSITIONAL[*]}"
    fi
fi

# ======================= 颜色定义 =======================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ======================= request_id =======================

# 同一轮所有 rank 必须同 ID, 跨轮绝不能复用 —— 见文件头注释。

# FT 指令会写 rendezvous 键, 必须显式给轮次号; chat/status/health/models 不需要。

FT_ROUND_COMMANDS=("pause" "retry" "scale_down")

needs_round() {
    local c
    for c in "${FT_ROUND_COMMANDS[@]}"; do
        [[ "$COMMAND" == "$c" ]] && return 0
    done
    return 1
}

if needs_round; then
    if [[ -n "$REQUEST_ID" && -n "$ROUND" ]]; then
        echo -e "${RED}错误: --request-id 和 --round 不能同时使用${NC}" >&2
        exit 1
    fi
    if [[ -z "$REQUEST_ID" ]]; then
        if [[ -z "$ROUND" ]]; then
            echo -e "${RED}错误: $COMMAND 需要 --round N (request_id = ft-${COMMAND}-round-N)${NC}" >&2
            echo -e "${RED}  跨节点部署: 每个节点都要传同一个 N 并同时执行, 每个新轮次换新的 N${NC}" >&2
            echo -e "${RED}  示例: bash api.sh $COMMAND --round 1${NC}" >&2
            exit 1
        fi
        if [[ ! "$ROUND" =~ ^[0-9]+$ ]]; then
            echo -e "${RED}错误: --round 必须是正整数, 收到: '$ROUND'${NC}" >&2
            exit 1
        fi
        REQUEST_ID="ft-${COMMAND}-round-${ROUND}"
    fi
fi

# ======================= 通用请求函数 =======================

do_chat() {
    local rank=$1
    local port=$((BASE_PORT + rank))
    local url="http://${HOST}:${port}/v1/chat/completions"

    local body
    body=$(cat <<ENDJSON
{
  "model": "/mnt/weight/DeepSeekV4-pro-0813-w4a8/",
  "messages": [{"role": "user", "content": "$PROMPT"}],
  "max_tokens": $MAX_TOKENS,
  "stream": false
}
ENDJSON
    )

    local start_time
    start_time=$(date +%s%N)

    local resp
    resp=$(curl -s -w "\n%{http_code}" --max-time "$TIMEOUT" \
        -X POST "$url" \
        -H "Content-Type: application/json" \
        -d "$body" 2>&1)

    local end_time
    end_time=$(date +%s%N)
    local elapsed_ms=$(( (end_time - start_time) / 1000000 ))

    local http_code
    http_code=$(echo "$resp" | tail -1)
    local body_resp
    body_resp=$(echo "$resp" | sed '$d')

    echo -e "${BOLD}━━━ Rank $rank (port $port) ━━━${NC}"

    if [[ "$http_code" == "200" ]]; then
        local content
        content=$(echo "$body_resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    c = d[\"choices\"][0][\"message\"][\"content\"]
    u = d.get(\"usage\", {})
    print(c)
    print(f\"---usage: prompt={u.get('prompt_tokens', '?')}, completion={u.get('completion_tokens', '?')}, total={u.get('total_tokens', '?')}\")
except Exception as e:
    print(f\"[parse error: {e}]\")
" 2>&1)
        echo -e "${GREEN}✓${NC} ${elapsed_ms}ms"
        echo "$content"
    else
        echo -e "${RED}✗ HTTP $http_code${NC} (${elapsed_ms}ms)"
        echo "$body_resp" | head -5
    fi
    echo ""
}

do_status() {
    local rank=$1
    local port=$((BASE_PORT + rank))
    local url="http://${HOST}:${port}/v1/fault_tolerance/status"

    local resp
    resp=$(curl -s -w "\n%{http_code}" --max-time "$TIMEOUT" "$url" 2>&1)
    local http_code
    http_code=$(echo "$resp" | tail -1)
    local body_resp
    body_resp=$(echo "$resp" | sed '$d')

    if [[ "$http_code" == "200" ]]; then
        local status_info
        status_info=$(echo "$body_resp" | python3 -m json.tool 2>/dev/null || echo "$body_resp")
        echo -e "Rank $rank (port $port): ${GREEN}✓${NC}"
        echo "$status_info"
    else
        echo -e "Rank $rank (port $port): ${RED}✗ HTTP $http_code${NC}"
        echo "  $body_resp" | head -3
    fi
}

# pause: 停止请求受理并进入 recovery, 供后续 retry / scale_down 使用。

# 只受理 HEALTHY 的引擎; 202 仅代表已接收, 真实结果靠 status 观察 (引擎状态
# 会从 healthy -> diagnosing -> unhealthy, 需要等当前 step 与设备任务排干)。

do_pause() {
    local rank=$1
    local port=$((BASE_PORT + rank))
    local url="http://${HOST}:${port}/v1/fault_tolerance/apply"

    local body='{"instruction": "pause", "params": {}, "request_id": "'"$REQUEST_ID"'"}'

    echo -e "Rank $rank (port $port): 发送 pause..."

    local resp
    resp=$(curl -s -w "\n%{http_code}" --max-time "$TIMEOUT" \
        -X POST "$url" \
        -H "Content-Type: application/json" \
        -d "$body" 2>&1)
    local http_code
    http_code=$(echo "$resp" | tail -1)
    local body_resp
    body_resp=$(echo "$resp" | sed '$d')

    if [[ "$http_code" == "202" ]]; then
        echo -e "Rank $rank (port $port): ${GREEN}✓ pause 已接受 (202), 收敛状态用 status 查看${NC}"
        echo "  $body_resp"
    else
        echo -e "Rank $rank (port $port): ${RED}✗ pause 失败 (HTTP $http_code)${NC}"
        echo "  $body_resp" | head -3
    fi
}

do_retry() {
    local rank=$1
    local port=$((BASE_PORT + rank))
    local url="http://${HOST}:${port}/v1/fault_tolerance/apply"

    local body='{"instruction": "retry", "params": {"timeout": '"$RETRY_TIMEOUT"'}, "request_id": "'"$REQUEST_ID"'"}'

    echo -e "Rank $rank (port $port): 发送 retry (timeout=${RETRY_TIMEOUT}s)..."

    local resp
    resp=$(curl -s -w "\n%{http_code}" --max-time "$((RETRY_TIMEOUT + 10))" \
        -X POST "$url" \
        -H "Content-Type: application/json" \
        -d "$body" 2>&1)
    local http_code
    http_code=$(echo "$resp" | tail -1)
    local body_resp
    body_resp=$(echo "$resp" | sed '$d')

    if [[ "$http_code" == "202" ]]; then
        echo -e "Rank $rank (port $port): ${GREEN}✓ retry 已接受 (202 async)${NC}"
        echo "  $body_resp"
    else
        echo -e "Rank $rank (port $port): ${RED}✗ retry 失败 (HTTP $http_code)${NC}"
        echo "  $body_resp" | head -3
    fi
}

do_health() {
    local rank=$1
    local port=$((BASE_PORT + rank))
    local url="http://${HOST}:${port}/health"

    local resp
    resp=$(curl -s -o /dev/null -w "%{http_code}" --max-time "$TIMEOUT" "$url" 2>&1)

    if [[ "$resp" == "200" ]]; then
        echo -e "Rank $rank (port $port): ${GREEN}✓ healthy${NC}"
    else
        echo -e "Rank $rank (port $port): ${RED}✗ HTTP $resp${NC}"
    fi
}

do_models() {
    local rank=$1
    local port=$((BASE_PORT + rank))
    local url="http://${HOST}:${port}/v1/models"

    local resp
    resp=$(curl -s -w "\n%{http_code}" --max-time "$TIMEOUT" "$url" 2>&1)
    local http_code
    http_code=$(echo "$resp" | tail -1)
    local body_resp
    body_resp=$(echo "$resp" | sed '$d')

    if [[ "$http_code" == "200" ]]; then
        local model_name
        model_name=$(echo "$body_resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    for m in d.get('data', []):
        print(f\"  {m['id']}\")
except Exception:
    print('  [parse error]')
" 2>&1)
        echo -e "Rank $rank (port $port): ${GREEN}✓${NC}"
        echo "$model_name"
    else
        echo -e "Rank $rank (port $port): ${RED}✗ HTTP $http_code${NC}"
    fi
}

do_scale_down() {
    local rank=$1
    local port=$((BASE_PORT + rank))
    local url="http://${HOST}:${port}/v1/fault_tolerance/apply"

    local params
    params="{\"removed_dp_ranks\": [$REMOVED_RANKS_JSON]"
    if [[ -n "$DP_STORE_PORT" ]]; then
        params+=", \"dp_store_port\": $DP_STORE_PORT"
    fi
    if [[ -n "$DP_MASTER_IP" ]]; then
        params+=", \"dp_master_ip\": \"$DP_MASTER_IP\""
    fi
    params+="}"

    local body="{\"instruction\": \"scale_down\", \"params\": $params, \"request_id\": \"$REQUEST_ID\"}"

    echo -e "Rank $rank (port $port): 发送 scale_down (移除 [$REMOVED_RANKS])..."

    local resp
    resp=$(curl -s -w "\n%{http_code}" --max-time "$((RETRY_TIMEOUT + 10))" \
        -X POST "$url" \
        -H "Content-Type: application/json" \
        -d "$body" 2>&1)
    local http_code
    http_code=$(echo "$resp" | tail -1)
    local body_resp
    body_resp=$(echo "$resp" | sed '$d')

    if [[ "$http_code" == "202" ]]; then
        echo -e "Rank $rank (port $port): ${GREEN}✓ scale_down 已接受 (202 async)${NC}"
        echo "  $body_resp"
    else
        echo -e "Rank $rank (port $port): ${RED}✗ scale_down 失败 (HTTP $http_code)${NC}"
        echo "  $body_resp" | head -3
    fi
}

# ======================= 执行 =======================

echo -e "${BOLD}============================================${NC}"
echo -e "${BOLD}  命令: $COMMAND | DP=$DP_SIZE | 端口: ${BASE_PORT}-$((BASE_PORT + DP_SIZE - 1))${NC}"
if needs_round; then
    echo -e "${BOLD}  request_id: ${CYAN}${REQUEST_ID}${NC}"
    echo -e "${BOLD}  跨节点要求: ${CYAN}所有节点同 request_id + 10s 内同时下发${NC}"
    case "$HOST" in
        0.0.0.0|127.0.0.1|localhost)
            echo -e "${YELLOW}  注意: HOST=$HOST 只打得到本节点的 rank, 其它节点须用同一个 --round 各自执行${NC}"
            ;;
    esac
fi
if [[ "$COMMAND" == "chat" ]]; then
    echo -e "${BOLD}  Prompt: ${CYAN}${PROMPT}${NC}"
fi
if [[ "$COMMAND" == "scale_down" ]]; then
    if [[ -z "$REMOVED_RANKS" ]]; then
        echo -e "${RED}错误: scale_down 需要指定移除的 rank, 如: bash api.sh scale_down 1,3${NC}"
        exit 1
    fi
    REMOVED_RANKS_JSON=$(echo "$REMOVED_RANKS" | tr ',' ', ')
    REMOVED_SET=",${REMOVED_RANKS},"

    # 移除 rank 0 时 TCPStore master 需重建, 必须指定新 master 的 ip 和 port

    if [[ "$REMOVED_SET" == *",0,"* ]]; then
        if [[ -z "$DP_STORE_PORT" || -z "$DP_MASTER_IP" ]]; then
            echo -e "${RED}错误: 移除 rank 0 时必须同时指定 --dp-store-port 和 --dp-master-ip${NC}"
            echo -e "${RED}  示例: bash api.sh scale_down 0 --dp-store-port 29501 --dp-master-ip 127.0.0.1${NC}"
            exit 1
        fi
    fi
    echo -e "${BOLD}  移除 rank: ${RED}[${REMOVED_RANKS}]${NC}"
    echo -e "${BOLD}  发送对象: ${CYAN}所有 $DP_SIZE 个 rank (含被移除的)${NC}"
fi
echo -e "${BOLD}============================================${NC}"
echo ""

dispatch() {
    local rank=$1
    case "$COMMAND" in
        chat)
            do_chat "$rank"
            ;;
        status)
            do_status "$rank"
            ;;
        pause)
            do_pause "$rank"
            ;;
        retry)
            do_retry "$rank"
            ;;
        health)
            do_health "$rank"
            ;;
        models)
            do_models "$rank"
            ;;
        scale_down)
            do_scale_down "$rank"
            ;;
    esac
}

if [[ "$SERIAL" == "true" ]]; then
    for ((R=0; R<DP_SIZE; R++)); do
        dispatch "$R"
    done
else
    PIDS=()
    TMPDIR_API=$(mktemp -d)
    trap 'rm -rf "$TMPDIR_API"' EXIT

    for ((R=0; R<DP_SIZE; R++)); do
        dispatch "$R" > "$TMPDIR_API/rank_$R.out" 2>&1 &
        PIDS+=($!)
    done

    for ((R=0; R<DP_SIZE; R++)); do
        wait "${PIDS[$R]}" 2>/dev/null || true
        cat "$TMPDIR_API/rank_$R.out"
    done

    rm -rf "$TMPDIR_API"
fi

echo -e "${BOLD}============================================${NC}"
echo -e "${BOLD}  完成${NC}"
echo -e "${BOLD}============================================${NC}"
