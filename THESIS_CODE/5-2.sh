# for training 
nohup python train_SinNet.py --epochs 2000000 --n_samples 1 --lr 0.01 &
# for drawing graph for evaluation of regression tasks
python test_SinNet.py 50 --models_dir BBP_SinNet2_models/theta_best.dat

