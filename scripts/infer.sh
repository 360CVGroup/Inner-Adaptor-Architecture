name="qihoo360/iaa-14-hf"
python -m iaa.eval.infer \
    --model-path $name \
    --image-path testimg/readpanda.jpg \
    --task_type MM \