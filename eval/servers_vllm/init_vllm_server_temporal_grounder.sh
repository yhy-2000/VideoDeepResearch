# ca video
export PATH="./envs/video/bin:$PATH"

declare -A GPU_CONFIG=(
    ["0"]="22345" 
)

model_name=/share/project/huaying/VideoDeepResearch/train/LLaMA-Factory-main/saves/qwen25-7b-sft-temporal_grounding_real_6k_base

NODE_NAME=$(hostname)

LOG_DIR="./VideoDeepResearch_release/eval/vllm_log"
mkdir -p "$LOG_DIR"

start_vllm_server() {
    local gpu_id=$1
    local port=$2
    local log_file="${LOG_DIR}/vllm_server_${NODE_NAME}_gpu${gpu_id}.log"
    local pid_file="${LOG_DIR}/vllm_server_${NODE_NAME}_gpu${gpu_id}.pid"
    local starting_flag_file="${LOG_DIR}/vllm_server_${NODE_NAME}_gpu${gpu_id}.starting"

    if [ -f "$starting_flag_file" ]; then
        local start_time=$(cat "$starting_flag_file")
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))

        if [ $elapsed -lt 300 ]; then
            echo "$(date): GPU ${gpu_id}正在启动中，跳过重复启动 (已启动 ${elapsed}s)" | tee -a "$log_file"
            return 0
        else
            echo "$(date): GPU ${gpu_id}启动超时，清理启动标记" | tee -a "$log_file"
            rm -f "$starting_flag_file"
        fi
    fi

    if check_server "$gpu_id"; then
        echo "$(date): GPU ${gpu_id}服务器已经在运行，跳过启动" | tee -a "$log_file"
        return 0
    fi

    if netstat -tlnp 2>/dev/null | grep -q ":${port}.*LISTEN"; then
        echo "$(date): 端口 ${port} 已被占用，无法启动GPU ${gpu_id}服务器" | tee -a "$log_file"
        return 1
    fi

    echo $(date +%s) > "$starting_flag_file"
    
    echo "$(date): 启动GPU ${gpu_id}上的VLLM服务器，端口：${port}..." | tee -a "$log_file"
    
    VLLM_LOGGING_LEVEL=DEBUG VLLM_USE_MODELSCOPE=false CUDA_VISIBLE_DEVICES=$gpu_id vllm serve $model_name \
        --port $port \
        --max-model-len 32768 \
        --tensor-parallel-size 1 \
        --enable-chunked-prefill \
        --gpu-memory-utilization 0.9 \
        >> "$log_file" 2>&1 &
    
    local server_pid=$!
    echo $server_pid > "$pid_file"
    echo "$(date): GPU ${gpu_id}服务器启动命令已执行 (PID: $server_pid, 端口: $port)" | tee -a "$log_file"

    local max_wait=200
    local wait_count=0
    while [ $wait_count -lt $max_wait ]; do
        if netstat -tlnp 2>/dev/null | grep -q ":${port}.*LISTEN"; then
            echo "$(date): GPU ${gpu_id}服务器启动成功 (端口: $port)" | tee -a "$log_file"
            rm -f "$starting_flag_file"
            return 0
        fi

        if ! ps -p "$server_pid" > /dev/null 2>&1; then
            echo "$(date): GPU ${gpu_id}服务器进程意外退出" | tee -a "$log_file"
            rm -f "$pid_file" "$starting_flag_file"
            return 1
        fi
        
        sleep 1
        wait_count=$((wait_count + 1))
    done
    
    echo "$(date): GPU ${gpu_id}服务器启动超时" | tee -a "$log_file"
    rm -f "$starting_flag_file"
    return 1
}

check_server() {
    local gpu_id=$1
    local port=${GPU_CONFIG[$gpu_id]}
    local pid_file="${LOG_DIR}/vllm_server_${NODE_NAME}_gpu${gpu_id}.pid"

    if [ ! -f "$pid_file" ]; then
        return 1
    fi
    
    local pid=$(cat "$pid_file")

    if ! ps -p "$pid" > /dev/null 2>&1; then
        echo "$(date): GPU ${gpu_id} PID ${pid} 进程不存在，清理PID文件" >> "${LOG_DIR}/vllm_server_${NODE_NAME}_gpu${gpu_id}.log"
        rm -f "$pid_file"
        return 1
    fi

    if ! ps -p "$pid" -o cmd --no-headers | grep -q "vllm serve"; then
        echo "$(date): GPU ${gpu_id} PID ${pid} 不是VLLM服务进程" >> "${LOG_DIR}/vllm_server_${NODE_NAME}_gpu${gpu_id}.log"
        rm -f "$pid_file"
        return 1
    fi

    if ! netstat -tlnp 2>/dev/null | grep -q ":${port}.*LISTEN"; then
        echo "$(date): GPU ${gpu_id} 端口 ${port} 未监听，但进程存在 - 可能正在初始化" >> "${LOG_DIR}/vllm_server_${NODE_NAME}_gpu${gpu_id}.log"

        return 0
    fi
    
    return 0
}

check_server_strict() {
    local gpu_id=$1
    local port=${GPU_CONFIG[$gpu_id]}
    local pid_file="${LOG_DIR}/vllm_server_${NODE_NAME}_gpu${gpu_id}.pid"
    local log_file="${LOG_DIR}/vllm_server_${NODE_NAME}_gpu${gpu_id}.log"

    if [ ! -f "$pid_file" ]; then
        echo "$(date): GPU ${gpu_id} PID文件不存在，需要启动服务器" >> "$log_file"
        return 1
    fi
    
    local pid=$(cat "$pid_file")

    if ! ps -p "$pid" > /dev/null 2>&1; then
        echo "$(date): GPU ${gpu_id} PID ${pid} 进程不存在，需要启动服务器" >> "$log_file"
        rm -f "$pid_file"
        return 1
    fi

    if ! ps -p "$pid" -o cmd --no-headers | grep -q "vllm serve"; then
        echo "$(date): GPU ${gpu_id} PID ${pid} 不是VLLM服务进程，需要重启" >> "$log_file"
        rm -f "$pid_file"
        return 1
    fi

    local port_check_count=0
    local max_port_checks=3
    
    for ((i=1; i<=max_port_checks; i++)); do
        if netstat -tlnp 2>/dev/null | grep -q ":${port}.*LISTEN"; then
            port_check_count=$((port_check_count + 1))
        fi
        sleep 1
    done
    
    if [ $port_check_count -eq 0 ]; then
        echo "$(date): GPU ${gpu_id} 端口 ${port} 连续 ${max_port_checks} 次检查都未监听，需要重启" >> "$log_file"
        return 1
    elif [ $port_check_count -lt $max_port_checks ]; then
        echo "$(date): GPU ${gpu_id} 端口 ${port} 检查 ${port_check_count}/${max_port_checks} 次成功，服务可能不稳定但暂不重启" >> "$log_file"
        return 0
    fi

    if command -v curl >/dev/null 2>&1; then
        local health_check_count=0
        local max_health_checks=2
        
        for ((i=1; i<=max_health_checks; i++)); do
            if curl -s --connect-timeout 5 "http://localhost:${port}/health" >/dev/null 2>&1; then
                health_check_count=$((health_check_count + 1))
            fi
            sleep 2
        done
        
        if [ $health_check_count -eq 0 ]; then
            echo "$(date): GPU ${gpu_id} 健康检查连续 ${max_health_checks} 次失败，但进程和端口正常，暂不重启" >> "$log_file"
            return 0
        fi
    fi
    
    echo "$(date): GPU ${gpu_id} 服务器状态正常，无需重启" >> "$log_file"
    return 0
}

stop_server() {
    local gpu_id=$1
    local pid_file="${LOG_DIR}/vllm_server_${NODE_NAME}_gpu${gpu_id}.pid"
    local log_file="${LOG_DIR}/vllm_server_${NODE_NAME}_gpu${gpu_id}.log"
    local starting_flag_file="${LOG_DIR}/vllm_server_${NODE_NAME}_gpu${gpu_id}.starting"

    rm -f "$starting_flag_file"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        echo "$(date): 停止GPU ${gpu_id}上的VLLM服务器 (PID: $pid)..." | tee -a "$log_file"

        kill "$pid" 2>/dev/null

        local wait_count=0
        while [ $wait_count -lt 10 ]; do
            if ! ps -p "$pid" > /dev/null 2>&1; then
                echo "$(date): GPU ${gpu_id}服务器已停止" | tee -a "$log_file"
                rm -f "$pid_file"
                return 0
            fi
            sleep 1
            wait_count=$((wait_count + 1))
        done

        echo "$(date): 强制终止GPU ${gpu_id}服务器..." | tee -a "$log_file"
        kill -9 "$pid" 2>/dev/null
        sleep 2
        
        rm -f "$pid_file"
    else
        echo "$(date): GPU ${gpu_id}服务器PID文件不存在" | tee -a "$log_file"
    fi

    local port=${GPU_CONFIG[$gpu_id]}
    local port_pid=$(netstat -tlnp 2>/dev/null | grep ":${port}.*LISTEN" | awk '{print $7}' | cut -d'/' -f1)
    if [ -n "$port_pid" ] && [ "$port_pid" != "-" ]; then
        echo "$(date): 端口 ${port} 仍被进程 ${port_pid} 占用，终止该进程" | tee -a "$log_file"
        kill -9 "$port_pid" 2>/dev/null
    fi
}

stop_all_servers() {
    echo "$(date): 停止所有VLLM服务器..."
    for gpu_id in "${!GPU_CONFIG[@]}"; do
        stop_server "$gpu_id"
    done

    rm -f "${LOG_DIR}"/vllm_server_${NODE_NAME}_gpu*.starting
}

cleanup() {
    echo "$(date): 收到退出信号，正在清理..."
    stop_all_servers
    exit 0
}

trap cleanup SIGINT SIGTERM

auto_restart() {
    
    local max_restart_attempts=100 
    local restart_delay=5

    declare -A restart_count
    declare -A consecutive_failures
    for gpu_id in "${!GPU_CONFIG[@]}"; do
        restart_count[$gpu_id]=0
        consecutive_failures[$gpu_id]=0
    done

    while true; do
        local all_running=true
        
        for gpu_id in "${!GPU_CONFIG[@]}"; do
            local port=${GPU_CONFIG[$gpu_id]}
            local log_file="${LOG_DIR}/vllm_server_${NODE_NAME}_gpu${gpu_id}.log"
            local starting_flag_file="${LOG_DIR}/vllm_server_${NODE_NAME}_gpu${gpu_id}.starting"

            if [ -f "$starting_flag_file" ]; then
                local start_time=$(cat "$starting_flag_file")
                local current_time=$(date +%s)
                local elapsed=$((current_time - start_time))
                
                if [ $elapsed -lt 300 ]; then
                    echo "$(date): GPU ${gpu_id}正在启动中，跳过检查 (已启动 ${elapsed}s)" | tee -a "$log_file"
                    continue
                else
                    echo "$(date): GPU ${gpu_id}启动超时，清理启动标记" | tee -a "$log_file"
                    rm -f "$starting_flag_file"
                fi
            fi

            if ! check_server "$gpu_id"; then
                consecutive_failures[$gpu_id]=$((${consecutive_failures[$gpu_id]} + 1))

                if [ ${consecutive_failures[$gpu_id]} -ge 3 ]; then
                    echo "$(date): GPU ${gpu_id}连续 ${consecutive_failures[$gpu_id]} 次检查失败，进行严格检查..." | tee -a "$log_file"

                    if check_server_strict "$gpu_id"; then
                        echo "$(date): GPU ${gpu_id}严格检查通过，服务器实际运行正常" | tee -a "$log_file"
                        consecutive_failures[$gpu_id]=0
                        continue
                    fi

                    if [ ${restart_count[$gpu_id]} -ge $max_restart_attempts ]; then
                        echo "$(date): GPU ${gpu_id}达到最大重启次数 ($max_restart_attempts)，停止重启" | tee -a "$log_file"
                        continue
                    fi

                    echo "$(date): GPU ${gpu_id}服务器确实需要重启... (尝试 $((${restart_count[$gpu_id]} + 1))/$max_restart_attempts)" | tee -a "$log_file"

                    stop_server "$gpu_id"
                    sleep 3

                    if start_vllm_server "$gpu_id" "$port"; then
                        echo "$(date): GPU ${gpu_id}服务器启动成功 (端口: $port)" | tee -a "$log_file"
                        restart_count[$gpu_id]=0
                        consecutive_failures[$gpu_id]=0
                    else
                        echo "$(date): GPU ${gpu_id}服务器启动失败" | tee -a "$log_file"
                        restart_count[$gpu_id]=$((${restart_count[$gpu_id]} + 1))
                        all_running=false
                    fi
                else
                    echo "$(date): GPU ${gpu_id}检查失败 (${consecutive_failures[$gpu_id]}/3)，等待更多失败后再重启" | tee -a "$log_file"
                    all_running=false
                fi
            else

                if [ ${consecutive_failures[$gpu_id]} -gt 0 ]; then
                    echo "$(date): GPU ${gpu_id}服务器恢复正常，重置失败计数" | tee -a "$log_file"
                    consecutive_failures[$gpu_id]=0
                fi

                if [ $(($(date +%s) % 300)) -eq 0 ]; then
                    echo "$(date): GPU ${gpu_id}服务器运行正常 (端口: $port)" | tee -a "$log_file"
                fi
            fi
        done

        if [ "$all_running" = true ]; then
            sleep 30
        else
            sleep $restart_delay
        fi
    done
}

case "$1" in
    start)
        echo "$(date): 手动启动所有VLLM服务器..."
        stop_all_servers
        for gpu_id in "${!GPU_CONFIG[@]}"; do
            port=${GPU_CONFIG[$gpu_id]}
            start_vllm_server "$gpu_id" "$port" &
        done
        wait
        echo "$(date): 所有服务器已启动"
        ;;
    stop)
        stop_all_servers
        echo "$(date): 所有服务器已停止"
        ;;
    restart)
        echo "$(date): 重启所有VLLM服务器..."
        stop_all_servers
        sleep 2
        for gpu_id in "${!GPU_CONFIG[@]}"; do
            port=${GPU_CONFIG[$gpu_id]}
            start_vllm_server "$gpu_id" "$port" &
        done
        wait
        echo "$(date): 所有服务器已重启"
        ;;
    status)
        echo "检查所有VLLM服务器状态："
        for gpu_id in "${!GPU_CONFIG[@]}"; do
            port=${GPU_CONFIG[$gpu_id]}
            starting_flag_file="${LOG_DIR}/vllm_server_${NODE_NAME}_gpu${gpu_id}.starting"
            
            if [ -f "$starting_flag_file" ]; then
                start_time=$(cat "$starting_flag_file")
                current_time=$(date +%s)
                elapsed=$((current_time - start_time))
                echo "  GPU ${gpu_id} (端口 ${port}): 正在启动中 (已启动 ${elapsed}s)"
            elif check_server "$gpu_id"; then
                pid_file="${LOG_DIR}/vllm_server_${NODE_NAME}_gpu${gpu_id}.pid"
                echo "  GPU ${gpu_id} (端口 ${port}): 正在运行 (PID: $(cat $pid_file))"
            else
                echo "  GPU ${gpu_id} (端口 ${port}): 未运行"
            fi
        done
        ;;
    detail)
        echo "检查所有VLLM服务器详细状态："
        for gpu_id in "${!GPU_CONFIG[@]}"; do
            port=${GPU_CONFIG[$gpu_id]}
            starting_flag_file="${LOG_DIR}/vllm_server_${NODE_NAME}_gpu${gpu_id}.starting"
            pid_file="${LOG_DIR}/vllm_server_${NODE_NAME}_gpu${gpu_id}.pid"
            
            echo "  GPU ${gpu_id} (端口 ${port}):"
            
            if [ -f "$starting_flag_file" ]; then
                start_time=$(cat "$starting_flag_file")
                current_time=$(date +%s)
                elapsed=$((current_time - start_time))
                echo "    状态: 正在启动中 (已启动 ${elapsed}s)"
            elif [ -f "$pid_file" ]; then
                local_pid=$(cat "$pid_file")
                if ps -p "$local_pid" > /dev/null 2>&1; then
                    echo "    状态: 进程运行中 (PID: $local_pid)"
                    if ps -p "$local_pid" -o cmd --no-headers | grep -q "vllm serve"; then
                        echo "    进程: VLLM服务进程 ✓"
                    else
                        echo "    进程: 非VLLM服务进程 ✗"
                    fi
                else
                    echo "    状态: PID文件存在但进程不存在 ✗"
                fi
            else
                echo "    状态: 未运行 (无PID文件)"
            fi

            if netstat -tlnp 2>/dev/null | grep -q ":${port}.*LISTEN"; then
                echo "    端口: ${port} 正在监听 ✓"
            else
                echo "    端口: ${port} 未监听 ✗"
            fi

            if command -v curl >/dev/null 2>&1; then
                if curl -s --connect-timeout 3 "http://localhost:${port}/health" >/dev/null 2>&1; then
                    echo "    健康检查: 通过 ✓"
                else
                    echo "    健康检查: 失败 ✗"
                fi
            else
                echo "    健康检查: 跳过 (curl不可用)"
            fi

            if check_server_strict "$gpu_id"; then
                echo "    综合状态: 健康 ✓"
            else
                echo "    综合状态: 需要重启 ✗"
            fi
            
            echo ""
        done
        ;;
    cleanup)
        echo "$(date): 清理所有残留文件和进程..."
        stop_all_servers

        rm -f "${LOG_DIR}"/vllm_server_${NODE_NAME}_gpu*.pid
        rm -f "${LOG_DIR}"/vllm_server_${NODE_NAME}_gpu*.starting

        echo "$(date): 查找并终止所有VLLM进程..."
        for gpu_id in "${!GPU_CONFIG[@]}"; do
            local port=${GPU_CONFIG[$gpu_id]}
            local vllm_pids=$(ps aux | grep "vllm serve" | grep "port $port" | grep -v grep | awk '{print $2}')
            if [ -n "$vllm_pids" ]; then
                echo "$(date): 终止GPU ${gpu_id}上的VLLM进程: $vllm_pids"
                kill -9 $vllm_pids 2>/dev/null
            fi
        done
        
        echo "$(date): 清理完成"
        ;;
    auto)
        echo "$(date): 启动自动重启模式..."
        auto_restart
        ;;
    *)
        echo "用法: $0 {start|stop|restart|status|detail|cleanup|auto}"
        echo "  start  - 启动所有服务器"
        echo "  stop   - 停止所有服务器"
        echo "  restart- 重启所有服务器"
        echo "  status - 检查所有服务器状态"
        echo "  detail - 检查所有服务器详细状态"
        echo "  cleanup- 清理所有残留文件和进程"
        echo "  auto   - 启动自动重启模式"
        echo ""
        echo "配置信息："
        for gpu_id in "${!GPU_CONFIG[@]}"; do
            port=${GPU_CONFIG[$gpu_id]}
            echo "  GPU ${gpu_id}: 端口 ${port}" &
        done
        wait
        echo ""
        echo "如果不提供参数，默认启动自动重启模式"
        echo ""
        auto_restart
        ;;
esac

    