!/bin/bash

# Successful run! Use this as the base line
python3 src/finetune_gpt2.py --model gpt2-medium --data_files ../NeuralNetworkStarter/data/clean/training_mix_data_2025_04_02.json --seq_length 128 --epochs 5 --learning_rate 1e-5 --batch_size 1 --gradient_steps 4 --weight_decay 0.02

# python3 src/finetune_gpt2.py --model gpt2-medium --data_files ../NeuralNetworkStarter/data/clean/training_mix_data_2025_04_02.json --seq_length 256 --epochs 3 --learning_rate 2e-5 --batch_size 1 --gradient_steps 4 --weight_decay 0.05 --warmup_ratio 0.1

# python3 src/finetune_gpt2.py --model gpt2-medium --data_files ../NeuralNetworkStarter/data/clean/training_mix_data_2025_04_02.json  --epochs 4 --learning_rate 3e-5 --batch_size 1 --gradient_steps 6 --seq_length 512