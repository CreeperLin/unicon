#!/usr/bin/env bash
"""
unicon_sim.sh

可编辑的全局变量:
	LOG_ROOT   - 日志根目录（例如 /home/caojiahang/Code/G1/logs）
	EXPERIMENT_NAME  - 实验分组目录（例如 g1_test）
	LOAD_RUN   - 实验名称（例如 1101-g123_feet0345）
	CHECKPOINT       - 迭代数（例如 30000），会用于 trace 文件名：trace_jit_CHECKPOINT${CHECKPOINT}.pt
	RT         - -rt 的值（例如 g1）

示例: 直接运行此脚本会使用脚本顶部的默认变量。
也可以通过环境变量覆盖，例如：
	LOG_ROOT=/some/path LOAD_RUN=my_exp ./unicon_sim.sh

该脚本会构造并执行原始的 python 命令：
	python -m unicon.run -imp <imp_path> -rt <RT> -m i -s unitree -dnstd -ssar -cmd vel -ncmd 9 -uspd -wi -scs <spec> -shmc -nowrp
"""

set -euo pipefail

# ------------------ 可修改的全局变量（用户请在此处修改） ------------------
# 设备类型：0=本地开发机（默认，LOG_ROOT=/home/.../Code/G1/logs），1=机器人（LOG_ROOT=/root/.../models）
DEVICE_TYPE=0

EXPERIMENT_NAME="g123rsw"
LOAD_RUN="1114-tune_on_1110-motion_1114"
CHECKPOINT=25000

# EXPERIMENT_NAME="g1_test"
# LOAD_RUN="1101-g123_feet0345"
# CHECKPOINT=30000


MODE="i"
SYSTEM="sims"
SIMS_SYSTEM='ig'


# 如果用户希望直接通过环境变量指定 LOG_ROOT，可在运行时传入 LOG_ROOT=/some/path
# 否则根据 DEVICE_TYPE 设置默认路径
: ${LOG_ROOT:=""}
if [[ -z "${LOG_ROOT}" ]]; then
	case ${DEVICE_TYPE} in 
	  0)
		LOG_ROOT="/home/caojiahang/Code/G1/logs"
		SPEC_ROOT="/home/caojiahang/Code/unicon"
		;;
	  1)
		LOG_ROOT="/root/GitRepo/cjh_deploy/models"
		SPEC_ROOT="/root/GitRepo/cjh_deploy/models"
		;;
	  *)
		echo "Error: Unknown DEVICE_TYPE '${DEVICE_TYPE}'" >&2
		exit 1
	esac
fi


EXTRA_ARGS=()
if [[ "${SYSTEM}" == "sims" ]]; then
	# "sim"
	# 将仿真专属参数添加到命令数组中
	EXTRA_ARGS+=(-nowrp -sst sims.systems."${SIMS_SYSTEM}")
else
	# "real"
	# 将真实设备专属参数添加到命令数组中
	EXTRA_ARGS+=(-wi -di none)
fi

case ${EXPERIMENT_NAME} in 
  g1_test)
	CMD_TYPE="vel"
	STATE_SPEC="imu2"
	RT="g1"
	EXTRA_ARGS+=(-uss)
	;;
  g1rsw)
  	CMD_TYPE="plan"
	STATE_SPEC="reach2"
	RT="g1reach"
	EXTRA_ARGS+=(-uss -pltp fix)
	;;
  g123rsw)
  	CMD_TYPE="plan"
	STATE_SPEC="reach2"
	RT="g123reach"
	EXTRA_ARGS+=(-uss -pltp fix)
	;;
  *)
	echo "Warning: Unknown EXPERIMENT_NAME '${EXPERIMENT_NAME}', using default STATE_SPEC."
esac

SPEC_PATH="/home/caojiahang/Code/unicon/${STATE_SPEC}_states_spec.yaml"

# 如果用户想直接指定整个 -imp 路径，可设置 IMP_PATH 环境变量或修改下面的构造逻辑
: ${IMP_PATH:="${LOG_ROOT}/${EXPERIMENT_NAME}/${LOAD_RUN}/trace_jit_iter${CHECKPOINT}.pt"}


if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
	usage
	exit 0
fi

# 检查 imp 文件是否存在，并提示但仍继续（保持与原命令一致）
if [[ ! -f "$IMP_PATH" ]]; then
	echo "Warning: imp file not found: $IMP_PATH" >&2
fi

# 构造命令数组以正确处理带空格的路径
CMD=(python -m unicon.run -imp "$IMP_PATH" -rt "$RT" -m "$MODE" -s "$SYSTEM" -dnstd -ssar -cmd "$CMD_TYPE" -ncmd 9 -uspd -scs "$SPEC_PATH" "${EXTRA_ARGS[@]}" -shmc)

echo "Execute:"
echo "  ${CMD[@]}"

# 使用 exec 替换进程（可选），便于信号传递；如果不想替换请改为 "${CMD[@]}" 执行
# 使用更通用的数组展开语法，避免 zsh-only 的 `${(@)CMD}` 在非 zsh 环境中报错
exec "${CMD[@]}"

