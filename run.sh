# train models
python codes/train.py --patch-rate 0.1 --data ./data --arch resnet50 --batch-size 512
python codes/train.py --patch-rate 0.1 --data ./data --arch resnet50 --batch-size 512 --resume ./data/models/resnet50_0.1/checkpoints/epoch_100_checkpoint.pth.tar --start-epoch 101

# analyze theoritical reuse rate
python codes/count_reuse.py

# image specific attack
python codes/image_specific_attck.py --arch resnet50 --ckpt data/models/resnet50_0.1/model_best.pth.tar --patch-rate 0.1 --batch-size 128

# universal attack
python codes/universal_attack.py --arch resnet50 --data ./data/train --ckpt ./data/models/resnet50_0.05/model_best.pth.tar

# find condidate positions
python codes/find_condidate.py

# detect
python codes/detect.py --arch resnet50 --patch-rate 0.1 --ckpt data/models/resnet50_0.1/model_best.pth.tar
