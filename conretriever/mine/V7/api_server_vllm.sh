python -m vllm.entrypoints.api_server \
    --host "0.0.0.0" \
    --port 30803 \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --trust-remote-code \
    --max-model-len 1024 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.90 \
    --dtype float32
    # --enforce-eager


说明：
--port 如果使用A800，端口必须是30001-30999
--tensor-parallel-size 与显卡数量保持一致
其余模型参数请参考：https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py 中的 EngineArgs 数据类

CUDA_VISIBLE_DEVICES=2,3,4,5 \
python -m vllm.entrypoints.api_server \
    --host "0.0.0.0" \
    --port 30803 \
    --model /remote-home/share/models/mistral_7b_instruct \
    --trust-remote-code \
    --max-model-len 1024 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.90 \
    --dtype float16
    # --enforce-eager