# prepare folders
mkdir -p data/models/resnet50_0.1/checkpoints/
mkdir -p data/attack_pic/
mkdir -p attack/ train/
ln -s /datasets/ImageNet_ILSVRC2012/ILSVRC2012_img_train train
ln -s /datasets/imagenet/convnet_ilsvrc12/ILSVRC2012_tar_file/val_fold val

# train models
python codes/train.py 
python codes/train.py 

# analyze theoritical reuse rate
python codes/count_reuse.py

# image specific attack
python codes/image_specific_attack.py 

# # universal attack
# python codes/universal_attack.py --arch resnet50 --data ./data/train --ckpt ./data/models/resnet50_0.05/model_best.pth.tar

# # find condidate positions
# python codes/find_condidate.py

# compute the adversarial examples attack accuracy
python compute_adv_acc.py

# detect
python codes/generate_mdr.py 

# using topk method to detect the adv patch
python codes/detect_using_topk.py
# using mrd method to detect the adv_patch
python codes/detect_using_mrd.py
