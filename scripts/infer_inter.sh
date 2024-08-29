name="qihoo360/iaa-14-hf"

python -m iaa.eval.infer_interleave \
    --model-path $INIT_MODEL_PATH/$name \
    --image-path testimg/COCO_train2014_000000014502.jpg \