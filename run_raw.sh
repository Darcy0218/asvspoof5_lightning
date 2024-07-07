# model=$1
# tlmodel=$2
# dmmodule=$3
# train_log_dir=$1
# lr=$2
# gpu=$3
# loss=$5


model=$(echo "$model" | sed 's/\//./g')
model=$(echo "$model" | sed 's/\.py//g')


if [ -z "$loss" ] || [ "$loss" = "-" ]; then
    loss="CE"
fi

# CUDA_VISIBLE_DEVICES=${gpu} 
line="
nohup python main.py 
--seed 1234
--module_model models.rawnet.RawNet2
--tl_model models.tl_model
--data_module utils.loadData.asvspoof_data_DA_still_process
--savedir a_log/rawnet
--optim_lr 0.0011
--gpuid 0
--batch_size 256
--epochs 100
--no_best_epochs 100
--optim adam
--weight_decay 0.0001
--loss WCE
--scheduler cosAnneal
--truncate 96000
> b_gpu_log/test_${gpu}.log
&"
# --usingDA
# --da_prob 0.7
echo ${line}
eval ${line}
