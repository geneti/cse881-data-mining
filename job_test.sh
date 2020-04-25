start=1
end=1
for i in $(seq $start $end);
    do
    model="blstm"

    # model
    mode="--mode train"
    data_path="--data_path ./src_zzc/data_ensemble/Training.txt"
    res_root="--res_root /mnt/gs18/scratch/users/chenzho7/cse881/res_ensemble"
    fold="--fold 5"
    # seed="--seed 1"

    # nn
    epoch="--epoch 20"
    batch_size="--batch_size 50"
    lr="--lr 0.01"
    desired_len_percent="--desired_len_percent 1.0"
    optimizer="--optimizer Adam"
    activate_func="--activate_func ReLU"

    # dan
    dan_hidden_sz="--dan_hidden_sz 100"
    dan_num_hidden="--dan_num_hidden 3"

    # embedding
    emb_drop_r="--emb_drop_r 0.5"
    emb_size="--emb_size 100"

    # LSTM
    lstm_drop_r="--lstm_drop_r 0.2"
    lstm_n_layer="--lstm_n_layer 1"
    lstm_hidden_sz="--lstm_hidden_sz 100"

    # cnn
    cnn_n_kernel="--cnn_n_kernel 100"
    cnn_kernel_sz="--cnn_kernel_sz 3"
    cnn_pool_kernel_sz="--cnn_pool_kernel_sz 2"

    command="$SCRATCH/venv/bin/python ./src_zzc/main.py $model --is_disk_limited --is_redirected $mode $data_path $res_root $fold $seed $epoch $batch_size $lr $desired_len_percent $optimizer $activate_func $dan_hidden_sz $dan_num_hidden $emb_drop_r $emb_size $lstm_drop_r $lstm_n_layer $lstm_hidden_sz $cnn_n_kernel $cnn_kernel_sz $cnn_pool_kernel_sz"

    echo "Running "
    echo $command
    echo ""
    echo ""

    eval $command

    echo ""
done

echo "Done"