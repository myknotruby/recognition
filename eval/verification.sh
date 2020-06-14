#CUDA_VISIBLE_DEVICES="7" python testimgs_verify.py --data-dir /train/mxnet-train/aligned_data --model /train/mxnet-train/insightface-master/model-r50-arc-lc/model,36  --target 'shebao' --batch-size 8
#CUDA_VISIBLE_DEVICES="7" python testimgs_verify.py --data-dir /train/mxnet-train/aligned_data --model /train/mxnet-train/face/insightface/models/model-r50-trip5e3/8702/model,6  --target 'ecu_clean' --batch-size 10

#CUDA_VISIBLE_DEVICES="7" python testimgs_verify.py --data-dir /train/mxnet-train/aligned_data --model /train/mxnet-train/face/insightface/neuron-selectivity-transfer/models/fmobilev3,2  --target 'ecu_clean' --batch-size 10

#CUDA_VISIBLE_DEVICES="5" python testimgs_verify.py --data-dir /train/mxnet-train/aligned_data --model /train/mxnet-train/face/insightface/recognition/models/r100-arcface-ela/r100-d512/model,69  --target 'guigang5k-test' --batch-size 1
#CUDA_VISIBLE_DEVICES="0" python testimgs_verify.py --data-dir /train/mxnet-train/aligned_data --model /train/mxnet-train/face/insightface/recognition/models/r100-arcface-ela/r100-d512/model,69  --target 'ecu_clean' --batch-size 1
#CUDA_VISIBLE_DEVICES="5" python testimgs_verify.py --data-dir /train/mxnet-train/aligned_data --model /train/mxnet-train/face/insightface/recognition/models/r200-arcface-egl/model,76  --target 'calfw-test' --batch-size 1
#CUDA_VISIBLE_DEVICES="4" python testimgs_verify.py --data-dir /train/mxnet-train/aligned_data --model /train/mxnet-train/face/insightface/recognition/models/r200-arcface-egl/model,104  --target 'cplfw-test' --batch-size 8
#CUDA_VISIBLE_DEVICES="1" python testimgs_verify.py --data-dir /train/mxnet-train/aligned_data --model /train/mxnet-train/face/insightface/models/model-r200-arc-emore/best/model,99  --target 'calfw-test' --batch-size 10
#CUDA_VISIBLE_DEVICES="4" python testimgs_verify2.py --data-dir /train/mxnet-train/aligned_data --model /train/mxnet-train/face/insightface/recognition/models/r200-arcface-africa/model,1  --target 'calfw-test' --batch-size 1
#CUDA_VISIBLE_DEVICES="3" python testimgs_verify.py --data-dir /train/mxnet-train/aligned_data --model modelto,0  --target 'calfw-test' --batch-size 10
#CUDA_VISIBLE_DEVICES="5" python testimgs_verify.py --data-dir /train/mxnet-train/aligned_data --model /train/mxnet-train/face/insightface/recognition/models/r200-arcface-africa/model,6  --target 'calfw-test' --batch-size 1
#python testimgs_verify.py --data-dir /train/mxnet-train/aligned_data --model /hdata/AlgOptimalLibrary/LiaoHuan/face-models/train-mxmodels/attention152/frvt-003/model,20  --target 'ecuador_retina,egy-test,africa-test' --batch-size 32 --gpu 3
#python testimgs_verify.py --data-dir /train/mxnet-train/aligned_data --model /hdata/AlgOptimalLibrary/LiaoHuan/face-models/face-mxmodels/feature/r100-69/model,69  --target 'ecuador_retina,egy-test,africa-test,guigang5k-test,southeastAsia-test' --batch-size 32 --gpu 3
#python testimgs_verify.py --data-dir /train/mxnet-train/aligned_data --model /hdata/AlgOptimalLibrary/LiaoHuan/face-models/face-mxmodels/feature/r100-69/model,69  --target 'southeastAsia-test' --batch-size 32 --gpu 3
#python testimgs_verify.py --data-dir /train/mxnet-train/aligned_data --model /hdata/AlgOptimalLibrary/LiaoHuan/face-models/face-mxmodels/feature/r50/model,18  --target 'southeastAsia-test' --batch-size 32 --gpu 2
#python testimgs_verify.py --data-dir /train/mxnet-train/aligned_data --model /hdata/AlgOptimalLibrary/LiaoHuan/face-models/train-mxmodels/attention152/rla/model,53  --target 'egy-test,africa-test,guigang5k-test,ecuador_retina,southeastAsia-test' --batch-size 32 --gpu 0
#python testimgs_verify.py --data-dir /train/mxnet-train/aligned_data --model /hdata/AlgOptimalLibrary/LiaoHuan/face-models/train-mxmodels/attention152/rla/model,53  --target 'zhongya-test' --batch-size 32 --gpu 0
#python testimgs_verify.py --data-dir /train/mxnet-train/aligned_data --model /hdata/AlgOptimalLibrary/LiaoHuan/face-models/train-mxmodels/attention152/frvt-003/model,20  --target 'zhongya-test' --batch-size 32 --gpu 0
#python testimgs_verify.py --data-dir /train/mxnet-train/aligned_data --model /hdata/AlgOptimalLibrary/LiaoHuan/face-models/face-mxmodels/feature/r50/model,18  --target 'zhongya-test' --batch-size 32 --gpu 0
#python testimgs_verify.py --data-dir /train/mxnet-train/aligned_data --model /train/mxnet-train/face/insightface/recognition/models/attse152-arcface-rlag-0.1-64/1k_fp16/model,43  --target 'ecuador_retina' --batch-size 32 --gpu 3
#python testimgs_verify.py --data-dir /train/mxnet-train/aligned_data --model /train/mxnet-train/face/insightface/recognition/models/attse152-arcface-rlag/best/model,57  --target 'guigang5k-test' --batch-size 32 --gpu 1
#python testimgs_verify.py --data-dir /train/mxnet-train/aligned_data --model /train/mxnet-train/face/insightface/recognition/models/attse152-arcface-rlag/best/model,57  --target 'zhongya-test' --batch-size 32 --gpu 2

###
python testimgs_verify.py --data-dir /train/mxnet-train/aligned_data --model /train/mxnet-train/face/insightface/recognition/models/r50-arcface-casia-EDUL/r50-arcface-casia/model,11  --target 'calfw-test' --batch-size 32 --gpu 0
python testimgs_verify.py --data-dir /train/mxnet-train/aligned_data --model /train/mxnet-train/face/insightface/recognition/models/r50-arcface-casia-EDUL2/r50-arcface-casia/model,11  --target 'calfw-test' --batch-size 32 --gpu 0
python testimgs_verify.py --data-dir /train/mxnet-train/aligned_data --model /train/mxnet-train/face/insightface/recognition/models/r50-arcface-casia-E/r50-arcface-casia/model,11  --target 'calfw-test' --batch-size 32 --gpu 1
