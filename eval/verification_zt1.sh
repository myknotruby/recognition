#CUDA_VISIBLE_DEVICES="7" python testimgs_verify.py --data-dir /train/mxnet-train/aligned_data --model /train/mxnet-train/insightface-master/model-r50-arc-lc/model,36  --target 'shebao' --batch-size 8
#CUDA_VISIBLE_DEVICES="0" python testimgs_verify.py --data-dir /train/mxnet-train/aligned_data --model /train/mxnet-train/face/insightface/models/model-r50-trip5e3/model,12  --target 'ecuador' --batch-size 4

#CUDA_VISIBLE_DEVICES="7" python testimgs_verify.py --data-dir /train/mxnet-train/aligned_data --model /train/mxnet-train/face/insightface/models/model-r50-trip5e3/8702/model,6  --target 'ecu_clean' --batch-size 10
CUDA_VISIBLE_DEVICES="1" python testimgs_verify.py --data-dir /train/mxnet-train/aligned_data --model /train/mxnet-train/face/insightface/recognition/models/r152-arcface-egll/model,155  --target 'ecu_clean' --batch-size 10
