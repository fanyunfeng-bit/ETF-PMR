## matlab
## CREMAD
#python main_ETF_matlab.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 120 --alpha 3.0 --train --momentum_coef 0.9 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1
#
## matlab init
#python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.95 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1
#
#
## larger alpha
#python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 4.0 --train --momentum_coef 0.95 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1
## other loss (dot regression)
#
#
## ETF_norm
## CREMAD
#python main_ETF_norm.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 150 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 120 --alpha 3.0 --train --temperature 30 --momentum_coef 0.9 --warmup_epoch 0


## balance
## CREMAD
## sum
#python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method sum --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.95 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1
#
## CREMAD
## gated
#python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method gated --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.95 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1
#
## CREMAD
## film
#python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method film --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.95 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1

## imbalance
## CREMAD
## concat
#python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.95 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1 --class_imbalance
#
## sum
#python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method sum --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.95 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1 --class_imbalance

## CREMAD
## gated
#python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method gated --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.9 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1 --class_imbalance
#
## CREMAD
## film
#python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method film --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.9 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1 --class_imbalance


# balance
# AVE
## concat
#python main_ETF_matlab_init.py --dataset AVE --modulation ETF --fusion_method concat --num_frame 4 --epochs 140 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 4 --train --momentum_coef 0.9 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1 --batch_size 32
#
## AVE
## sum
#python main_ETF_matlab_init.py --dataset AVE --modulation ETF --fusion_method sum --num_frame 4 --epochs 140 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 4 --train --momentum_coef 0.9 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1 --batch_size 32
#
## AVE
## gated
#python main_ETF_matlab_init.py --dataset AVE --modulation ETF --fusion_method gated --num_frame 4 --epochs 140 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 4 --train --momentum_coef 0.9 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1 --batch_size 32
#
## AVE
## film
#python main_ETF_matlab_init.py --dataset AVE --modulation ETF --fusion_method film --num_frame 4 --epochs 140 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 4 --train --momentum_coef 0.9 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1 --batch_size 32
#
#
## imbalance
## AVE
## concat
#python main_ETF_matlab_init.py --dataset AVE --modulation ETF --fusion_method concat --num_frame 4 --epochs 140 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 4 --train --momentum_coef 0.9 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1 --class_imbalance --batch_size 32
#
## AVE
## sum
#python main_ETF_matlab_init.py --dataset AVE --modulation ETF --fusion_method sum --num_frame 4 --epochs 140 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 4 --train --momentum_coef 0.9 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1 --class_imbalance --batch_size 32
#
## AVE
## gated
#python main_ETF_matlab_init.py --dataset AVE --modulation ETF --fusion_method gated --num_frame 4 --epochs 140 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 4 --train --momentum_coef 0.9 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1 --class_imbalance --batch_size 32
#
## AVE
## film
#python main_ETF_matlab_init.py --dataset AVE --modulation ETF --fusion_method film --num_frame 4 --epochs 140 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 4 --train --momentum_coef 0.9 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1 --class_imbalance --batch_size 32



## optimizer
## CREMAD
## concat
#python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.90 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1 --optimizer AdaGrad
#python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.90 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1 --optimizer Adam
#python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.90 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1 --class_imbalance --optimizer AdaGrad
#python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.90 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1 --class_imbalance --optimizer Adam
#
## AVE concat
#python main_ETF_matlab_init.py --dataset AVE --modulation ETF --fusion_method concat --num_frame 4 --epochs 140 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 4 --train --momentum_coef 0.9 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1 --batch_size 32 --optimizer AdaGrad
#python main_ETF_matlab_init.py --dataset AVE --modulation ETF --fusion_method concat --num_frame 4 --epochs 140 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 4 --train --momentum_coef 0.9 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1 --batch_size 32 --optimizer Adam
#python main_ETF_matlab_init.py --dataset AVE --modulation ETF --fusion_method concat --num_frame 4 --epochs 140 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 4 --train --momentum_coef 0.9 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1 --class_imbalance --batch_size 32 --optimizer AdaGrad
#python main_ETF_matlab_init.py --dataset AVE --modulation ETF --fusion_method concat --num_frame 4 --epochs 140 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 4 --train --momentum_coef 0.9 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1 --class_imbalance --batch_size 32 --optimizer Adam



## momentum
#python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.98 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1
#python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.95 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1
#python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.90 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1
#python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.85 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1
#python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.80 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1
#python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.70 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1
##python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.50 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1
##python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.40 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1
##python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.30 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1
#
#python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.98 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1 --class_imbalance
##python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.95 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1 --class_imbalance
##python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.90 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1 --class_imbalance
##python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.85 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1 --class_imbalance
#python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.80 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1 --class_imbalance
#python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.70 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1 --class_imbalance
##python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.50 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1 --class_imbalance
##python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.40 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1 --class_imbalance
##python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.30 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1 --class_imbalance

#python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.60 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1
#python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.50 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1
##python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.40 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1
#python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.30 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1
##python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.20 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1
#python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.10 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1
#
#python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.60 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1 --class_imbalance
#python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.50 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1 --class_imbalance
#python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.30 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1 --class_imbalance
#python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.10 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1 --class_imbalance


#python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 1.0 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1
#python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 1.0 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1 --class_imbalance
#python main_ETF_matlab_init.py --dataset AVE --modulation ETF --fusion_method concat --num_frame 4 --epochs 140 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 4 --train --momentum_coef 1.0 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1 --class_imbalance --batch_size 32
#python main_ETF_matlab_init.py --dataset AVE --modulation ETF --fusion_method concat --num_frame 4 --epochs 140 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 4 --train --momentum_coef 1.0 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1 --batch_size 32


#python main_ETF_matlab_init.py --dataset AVE --modulation ETF --fusion_method concat --num_frame 4 --epochs 140 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 4 --train --momentum_coef 0.9 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1 --batch_size 32
#python main_ETF_matlab_init.py --dataset AVE --modulation ETF --fusion_method concat --num_frame 4 --epochs 140 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 4 --train --momentum_coef 0.8 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1 --batch_size 32

#
#python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.7 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1
#python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.7 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1 --class_imbalance
#python main_ETF_matlab_init.py --dataset AVE --modulation ETF --fusion_method concat --num_frame 4 --epochs 140 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 4 --train --momentum_coef 0.9 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1 --class_imbalance --batch_size 32
#python main_ETF_matlab_init.py --dataset AVE --modulation ETF --fusion_method concat --num_frame 4 --epochs 140 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 4 --train --momentum_coef 0.9 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1 --batch_size 32

#python generate_feature.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --momentum_coef 0.9 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1 --ckpt_path ckpt/ETF-matlab-init/model-CREMAD/dist-optim_False/train_log-alpha3.0-lr0.001-fusion_concat-CI_False-optim_SGD-Mom_0.7-AP5-freq1-coef1-0.9/save_model.pt



#python main_ETF_matlab.py --dataset AVE --modulation ETF --fusion_method concat --num_frame 4 --epochs 140 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 4 --train --momentum_coef 0.9 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1 --class_imbalance --batch_size 32


#python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.7 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1
python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.7 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 2
python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.7 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 4
python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.7 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 8
python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.7 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 10
python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.7 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 16

python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.7 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 2 --class_imbalance
python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.7 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 4 --class_imbalance
python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.7 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 8 --class_imbalance
python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.7 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 10 --class_imbalance
python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method concat --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.7 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 16 --class_imbalance

