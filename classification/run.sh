set -x
for MODEL in resnet18 resnet50 inceptionv3 mobilenetv2_w1 shufflenet_g1_w1 sqnxt23_w2
do
	echo Testing $MODEL ...
	python uniform_test.py 		\
		--dataset=imagenet 		\
		--model=$MODEL 			\
		--batch_size=64 		\
		--test_batch_size=512
	python mixed_precision_test.py \
		--dataset=imagenet \
		--model=inceptionv3 \
		--batch_size=16 \
		--test_batch_size=16 \
done

