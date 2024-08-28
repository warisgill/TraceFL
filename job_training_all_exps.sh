# # srun --ntasks=16 python -m tracefl.main

# key_start='For-Main-Training-'
# num_clients=100
# clients_per_round=10
# dirichlet_alpha="0.1,0.2,0.3"


# echo "      ****************** Image Classification Experiments ******************"

# img_models="resnet18,densenet121"
# img_datasets="mnist,cifar10"
# num_rounds=50
# python -m tracefl.main --multirun do_training=True exp_key=$key_start model.name=$img_models dataset.name=$img_datasets num_clients=$num_clients num_rounds=$num_rounds clients_per_round=$clients_per_round dirichlet_alpha=$dirichlet_alpha dry_run=False | tee -a logs/train_image_classification_exp.log



# echo "      ****************** Medical Image Classification Experiments ******************"

# img_models="resnet18,densenet121"
# img_datasets="pathmnist,organamnist"
# num_rounds=25
# python -m tracefl.main --multirun do_training=True exp_key=$key_start model.name=$img_models dataset.name=$img_datasets num_clients=$num_clients num_rounds=$num_rounds clients_per_round=$clients_per_round dirichlet_alpha=$dirichlet_alpha dry_run=False | tee -a logs/train_image_classification_exp.log



# echo "      ****************** Text Classification Experiments ******************"
# text_models="openai-community/openai-gpt,google-bert/bert-base-cased"
# text_dataset="dbpedia_14,yahoo_answers_topics"
# num_rounds=25

# python -m tracefl.main --multirun do_training=True exp_key=$key_start model.name=$text_models dataset.name=$text_dataset num_clients=$num_clients num_rounds=$num_rounds clients_per_round=$clients_per_round dirichlet_alpha=$dirichlet_alpha dry_run=False | tee -a logs/train_text_classification_exp.log



# echo "      ****************** Shard Per Label Experiments ******************"

# # bash job_sharded_partition.sh                                                                                                                                                                             




# echo "      ****************** Differential Privacy Experiments ******************"

# num_rounds=15
# dp_noise="0.0001,0.0003,0.0007,0.0009,0.001,0.003" 
# dp_clip="15,50"
# num_clients=100
# clients_per_round=10


# # densenet on mnist
# python -m tracefl.main --multirun do_training=True exp_key=DP-image model.name="densenet121" dataset.name="mnist" num_clients=$num_clients num_rounds=$num_rounds clients_per_round=$clients_per_round dirichlet_alpha=0.3 noise_multiplier=$dp_noise clipping_norm=$dp_clip dry_run=False | tee -a logs/train_dp_exp_image.log

# python -m tracefl.main --multirun do_training=True exp_key=DP-image model.name="densenet121" dataset.name="pathmnist" num_clients=$num_clients num_rounds=$num_rounds clients_per_round=$clients_per_round dirichlet_alpha=0.3 noise_multiplier=$dp_noise clipping_norm=$dp_clip dry_run=False | tee -a logs/train_dp_exp_image.log

# python -m tracefl.main --multirun do_training=True exp_key=DP-text model.name="openai-community/openai-gpt" dataset.name="dbpedia_14" num_clients=$num_clients num_rounds=$num_rounds clients_per_round=$clients_per_round dirichlet_alpha=0.3 noise_multiplier=$dp_noise clipping_norm=$dp_clip dry_run=False | tee -a logs/train_dp_exp_text.log



echo "      ****************** 2 Differential Privacy Experiments ******************"

num_rounds=15
# dp_noise="0.0001,0.0003,0.0007,0.0009,0.001,0.003" 
# dp_clip="15,50"
num_clients=100
clients_per_round=10

dp_noise=0.003
dp_clip=15
python -m tracefl.main --multirun do_training=True exp_key=DP-text model.name="openai-community/openai-gpt,google-bert/bert-base-cased" dataset.name="dbpedia_14" num_clients=$num_clients num_rounds=$num_rounds clients_per_round=$clients_per_round dirichlet_alpha=0.3 noise_multiplier=$dp_noise clipping_norm=$dp_clip dry_run=False | tee -a logs/train_dp_exp_text2.log

dp_noise=0.006
dp_clip=10

python -m tracefl.main --multirun do_training=True exp_key=DP-text model.name="openai-community/openai-gpt,google-bert/bert-base-cased" dataset.name="dbpedia_14" num_clients=$num_clients num_rounds=$num_rounds clients_per_round=$clients_per_round dirichlet_alpha=0.3 noise_multiplier=$dp_noise clipping_norm=$dp_clip dry_run=False | tee -a logs/train_dp_exp_text2.log

dp_noise=0.012
dp_clip=15

python -m tracefl.main --multirun do_training=True exp_key=DP-text model.name="openai-community/openai-gpt,google-bert/bert-base-cased" dataset.name="dbpedia_14" num_clients=$num_clients num_rounds=$num_rounds clients_per_round=$clients_per_round dirichlet_alpha=0.3 noise_multiplier=$dp_noise clipping_norm=$dp_clip dry_run=False | tee -a logs/train_dp_exp_text2.log




# echo "      ****************** Scalablity Experiments ******************"
# num_rounds=15

# scaling_clients="200,400,600,800,1000"
# python -m tracefl.main --multirun do_training=True exp_key=Scaling model.name="openai-community/openai-gpt" dataset.name="dbpedia_14" num_clients=$scaling_clients num_rounds=$num_rounds clients_per_round=10 dirichlet_alpha=0.3 dry_run=False | tee -a logs/train_scaling_exp_text_total_clients.log


# per_round_clients="20,30,40,50"
# python -m tracefl.main --multirun do_training=True exp_key=Scaling model.name="openai-community/openai-gpt" dataset.name="dbpedia_14" num_clients=400 num_rounds=$num_rounds clients_per_round=$per_round_clients dirichlet_alpha=0.3 dry_run=False | tee -a logs/train_scaling_exp_text_clients_per_round.log

# # scalablity with number of rounds
# python -m tracefl.main --multirun do_training=True exp_key=Scaling model.name="openai-community/openai-gpt" dataset.name="dbpedia_14" num_clients=400 num_rounds=100 clients_per_round=10 dirichlet_alpha=0.3 dry_run=False | tee -a logs/train_scaling_exp_num_rounds.log




echo "*** Dirchlet Alpha Experiments ***"
num_clients=100
num_rounds=15
dirichlet_alpha="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1"
python -m tracefl.main --multirun do_training=True exp_key=Dirichlet-Alpha model.name="openai-community/openai-gpt" dataset.name="yahoo_answers_topics,dbpedia_14" num_clients=$num_clients num_rounds=$num_rounds clients_per_round=10 dirichlet_alpha=$dirichlet_alpha dry_run=False | tee -a logs/train_dirichlet_alpha.log
# python -m tracefl.main --multirun do_training=True exp_key=Dirichlet-Alpha model.name="densenet121" dataset.name="pathmnist,organamnist,mnist,cifar10" num_clients=$num_clients num_rounds=$num_rounds clients_per_round=10 dirichlet_alpha=$dirichlet_alpha dry_run=False  | tee -a logs/train_dirichlet_alpha.log




# bash job_prov.sh

