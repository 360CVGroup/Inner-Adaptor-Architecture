
save_root="./refcoco_re/"
mkdir $save_root


INIT_MODEL_PATH="qihoo360/"
for SPLIT in {"REC_refcoco_unc_testA","REC_refcoco_unc_testB","REC_refcoco_unc_val",}
do
for name in {"iaa-14-hf",}
do


if [ -f "./refcoco_re/iaa-14-hf.txt" ];then
    echo "The file exists"
else
    touch $save_root/$name.txt
fi

torchrun --master_port=7777 --nproc_per_node 8 -m iaa.eval.model_vqa_refcoco_llama3 \
    --model-path $INIT_MODEL_PATH/$name \
    --question-file ./refcoco/$SPLIT.jsonl \
    --image-folder COCO/train2014 \
    --answers-file ./refcoco_re/$name/$SPLIT.json \
    --temperature 0 \

python ./iaa/eval/compute_precision.py ./refcoco_re/$name/$SPLIT.json ./refcoco/$SPLIT.jsonl $save_root/$name.txt


done
done
