
name="qihoo360/iaa-14-hf"

torchrun --master_port=7777 --nproc_per_node 8 -m iaa.eval.model_vqa_loader_llama3 \
    --model-path $name \
    --question-file ./MME/llava_mme.jsonl \
    --image-folder ./MME/MME_Benchmark_release_version \
    --answers-file ./MME/answers/$name.jsonl \
    --temperature 0 \

cd ./MME

python convert_answer_to_mme.py --experiment $name

cd eval_tool

python calculation.py --results_dir answers/$name
