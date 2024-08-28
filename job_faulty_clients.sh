parallel_processes=3
# check_prov_cache=False
num_rounds=15
num_clients=10
clients_per_round=10
check_prov_cache=False


echo "      ****************** Medical Image Faulty Clients Experiments ******************"
img_models="densenet121,resnet18"

key_start="med_image_faulty"
faulty_clients_ids=["8"]
## key_start="med_image_non_faulty"
## faulty_clients_ids=[]
## label2flip={}

img_datasets="pathmnist"
label2flip={8:7,0:7,1:7} 
python -m tracefl.main --multirun data_dist.dist_type='PathologicalPartitioner-3' label2flip=$label2flip noise_rate=$label2flip  faulty_clients_ids=$faulty_clients_ids do_training=True exp_key=$key_start model.name=$img_models dataset.name=$img_datasets num_clients=$num_clients num_rounds=$num_rounds clients_per_round=$clients_per_round  dry_run=False | tee -a logs/train_pathmnist.log

# provenance
max_server_data_size=2048
python -m tracefl.main --multirun data_dist.dist_type='PathologicalPartitioner-3' label2flip=$label2flip  noise_rate=$label2flip faulty_clients_ids=$faulty_clients_ids do_training=False exp_key=$key_start model.name=$img_models dataset.name=$img_datasets num_clients=$num_clients num_rounds=$num_rounds clients_per_round=$clients_per_round  dry_run=False check_prov_cache=$check_prov_cache parallel_processes=$parallel_processes do_provenance=True data_dist.max_server_data_size=$max_server_data_size | tee  logs/prov_pathological_pathmnist.log



img_datasets="organamnist"
faulty_clients_ids=["0"]
label2flip={0:5,1:5,2:5} 
python -m tracefl.main --multirun data_dist.dist_type='PathologicalPartitioner-3' label2flip=$label2flip noise_rate=$label2flip  faulty_clients_ids=$faulty_clients_ids do_training=True exp_key=$key_start model.name=$img_models dataset.name=$img_datasets num_clients=$num_clients num_rounds=$num_rounds clients_per_round=$clients_per_round  dry_run=False | tee -a logs/train_image_faulty.log

# provenance
max_server_data_size=2048
python -m tracefl.main --multirun data_dist.dist_type='PathologicalPartitioner-3' label2flip=$label2flip  noise_rate=$label2flip faulty_clients_ids=$faulty_clients_ids do_training=True exp_key=$key_start model.name=$img_models dataset.name=$img_datasets num_clients=$num_clients num_rounds=$num_rounds clients_per_round=$clients_per_round  dry_run=False check_prov_cache=$check_prov_cache parallel_processes=$parallel_processes do_provenance=True data_dist.max_server_data_size=$max_server_data_size hydra/launcher=joblib  | tee -a logs/train_image_faulty.log

# key_start="faulty"
# img_datasets="cifar10,mnist"
# faulty_clients_ids=["0"]
# label2flip={0:5,1:5,2:5} 
# python -m tracefl.main --multirun data_dist.dist_type='PathologicalPartitioner-3' label2flip=$label2flip noise_rate=$label2flip  faulty_clients_ids=$faulty_clients_ids do_training=True exp_key=$key_start model.name=$img_models dataset.name=$img_datasets num_clients=$num_clients num_rounds=$num_rounds clients_per_round=$clients_per_round  dry_run=False | tee -a logs/train_image_faulty.log

# # provenance
# max_server_data_size=2048
# python -m tracefl.main --multirun data_dist.dist_type='PathologicalPartitioner-3' label2flip=$label2flip  noise_rate=$label2flip faulty_clients_ids=$faulty_clients_ids do_training=True exp_key=$key_start model.name=$img_models dataset.name=$img_datasets num_clients=$num_clients num_rounds=$num_rounds clients_per_round=$clients_per_round  dry_run=False check_prov_cache=$check_prov_cache parallel_processes=$parallel_processes do_provenance=True data_dist.max_server_data_size=$max_server_data_size hydra/launcher=joblib  | tee -a logs/train_image_faulty.log





echo "      ****************** Text Faulty Clients Experiments ******************"
check_prov_cache=True
text_models="openai-community/openai-gpt"
text_dataset="dbpedia_14,yahoo_answers_topics"
faulty_clients_ids=["0"]
label2flip={0:5,1:5,2:5} 
python -m tracefl.main --multirun data_dist.dist_type='PathologicalPartitioner-3' label2flip=$label2flip noise_rate=$label2flip  faulty_clients_ids=$faulty_clients_ids do_training=True exp_key=$key_start model.name=$text_models dataset.name=$text_dataset num_clients=$num_clients num_rounds=$num_rounds clients_per_round=$clients_per_round  dry_run=False | tee -a logs/train_image_faulty.log

# # provenance
max_server_data_size=2048
python -m tracefl.main --multirun data_dist.dist_type='PathologicalPartitioner-3' label2flip=$label2flip  noise_rate=$label2flip faulty_clients_ids=$faulty_clients_ids do_training=True exp_key=$key_start model.name=$text_models dataset.name=$text_dataset num_clients=$num_clients num_rounds=$num_rounds clients_per_round=$clients_per_round  dry_run=False check_prov_cache=$check_prov_cache parallel_processes=$parallel_processes do_provenance=True data_dist.max_server_data_size=$max_server_data_size hydra/launcher=joblib  | tee -a logs/train_image_faulty.log







