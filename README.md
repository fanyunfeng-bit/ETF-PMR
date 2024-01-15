## Download data
- CREMAD: `https://github.com/CheyneyComputerScience/CREMA-D`
- AVE: `https://sites.google.com/view/audiovisualresearch`
- CG-MNIST: `https://drive.google.com/file/d/1RVMRtN3C5MWUI9pYyZRvifLLJZVn6H_T/view?usp=drive_link`


## Preprocess data
- `video_preprocessing.py` in `./data/CREMAD` for CREMAD.
- `video_preprocessing.py` in `./data/AVE` for AVE.

## Split dataset to clients
- CREMAD and AVE: `preprocess.py` in `./data`.

## train

- CREMAD: \
`python main_ETF_matlab_init.py --dataset CREMAD --modulation ETF --fusion_method sum --num_frame 3 --epochs 120 --learning_rate 0.001 --modulation_starts 0 --modulation_ends 80 --alpha 3.0 --train --momentum_coef 0.9 --warmup_epoch 0 --adaptive_epoch 5 --proto_update_freq 1`
