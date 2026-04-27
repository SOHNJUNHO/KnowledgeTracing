FROM vllm/vllm-openai:latest

# Bake in the flags tuned on the GCE spot VM.
# Adjust --quantization, --max-num-seqs, --gpu-memory-utilization
# based on your benchmarking results before pushing to Cloud Run.
CMD ["--model", "Qwen/Qwen2.5-7B-Instruct", \
     "--quantization", "awq", \
     "--max-num-seqs", "32", \
     "--gpu-memory-utilization", "0.85", \
     "--port", "8080"]
