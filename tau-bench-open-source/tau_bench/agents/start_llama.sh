vllm serve meta-llama/Llama-3.3-70B-Instruct \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json \
    --chat-template tau_bench/agents/llama_tool_calling_template.jinja \
    --tensor-parallel-size 2 \
    --port 8000

## Added chunked prefill
vllm serve meta-llama/Llama-3.3-70B-Instruct --enable-auto-tool-choice --tool-call-parser llama3_json --chat-template tau-bench-open-source/tau_bench/agents/llama_tool_calling_template.jinja --tensor-parallel-size 2 --gpu-memory-utilization=0.96 --max-model-len 22000 --enable-chunked-prefill