set -x

python3 mixed_precision_test.py \
    --dataset=imagenet \
    --model=resnet18 \
    --batch_size=64 \
    --test_batch_size=256 \

python3 mixed_precision_test.py \
    --dataset=imagenet \
    --model=resnet50 \
    --batch_size=64 \
    --test_batch_size=256 \

python3 mixed_precision_test.py \
    --dataset=imagenet \
    --model=mobilenetv2_w1 \
    --batch_size=64 \
    --test_batch_size=256 \

python3 mixed_precision_test.py \
    --dataset=imagenet \
    --model=shufflenet_g1_w1 \
    --batch_size=32 \
    --test_batch_size=256 \

python3 mixed_precision_test.py \
    --dataset=imagenet \
    --model=sqnxt23_w2 \
    --batch_size=64 \
    --test_batch_size=512 \

python3 mixed_precision_test.py \
    --dataset=imagenet \
    --model=inceptionv3 \
    --batch_size=16 \
    --test_batch_size=32 \
