# Function to assert equality
assert_equal() {
    if [ "$1" = "$2" ]; then
        echo "Assertion passed: $1 equals $2"
    else
        echo "Assertion failed: $1 does not equal $2"
    fi
}

key_start='Sharded-Partition-'
num_clients=5
clients_per_round=5
dist_type="sharded-non-iid-2"
dirichlet_alpha=-1


assert_equal $num_clients $clients_per_round

echo "      ****************** Image Classification Experiments ******************"
img_models="densenet121"
img_datasets="mnist,organamnist"
num_rounds=25
python -m tracefl.main --multirun do_training=True exp_key=$key_start model.name=$img_models dataset.name=$img_datasets num_clients=$num_clients num_rounds=$num_rounds clients_per_round=$clients_per_round dirichlet_alpha=$dirichlet_alpha  data_dist.max_per_client_data_size=-1 data_dist.dist_type=$dist_type dry_run=False | tee -a logs/train_sharded_image_classification_exp.log
python -m tracefl.main --multirun do_training=True exp_key=$key_start model.name=$img_models dataset.name=$img_datasets num_clients=$num_clients num_rounds=$num_rounds clients_per_round=$clients_per_round dirichlet_alpha=$dirichlet_alpha  data_dist.max_per_client_data_size=-1 data_dist.dist_type=$dist_type dry_run=False | tee -a logs/train_sharded_med_image_classification_exp.log


img_datasets="mnist"
num_clients=10
clients_per_round=10
dist_type="sharded-non-iid-1"

python -m tracefl.main --multirun do_training=True exp_key=$key_start model.name=$img_models dataset.name=$img_datasets num_clients=$num_clients num_rounds=$num_rounds clients_per_round=$clients_per_round dirichlet_alpha=$dirichlet_alpha  data_dist.max_per_client_data_size=-1 data_dist.dist_type=$dist_type dry_run=False | tee -a logs/train_sharded_image_classification_exp.log


img_datasets="organamnist"
num_clients=11
clients_per_round=11
dist_type="sharded-non-iid-1"
python -m tracefl.main --multirun do_training=True exp_key=$key_start model.name=$img_models dataset.name=$img_datasets num_clients=$num_clients num_rounds=$num_rounds clients_per_round=$clients_per_round dirichlet_alpha=$dirichlet_alpha  data_dist.max_per_client_data_size=-1 data_dist.dist_type=$dist_type dry_run=False | tee -a logs/train_sharded_med_image_classification_exp.log








# echo "      ****************** Text Classification Experiments ******************"

# text_models="openai-community/openai-gpt"
# text_dataset="dbpedia_14"
# num_rounds=25
# num_clients=14
# clients_per_round=14
# dist_type="pathological-non-iid-1"

# # assert_equal $num_clients $clients_per_round
# # python -m tracefl.main --multirun do_training=True exp_key=$key_start model.name=$text_models dataset.name=$text_dataset num_clients=$num_clients num_rounds=$num_rounds clients_per_round=$clients_per_round dirichlet_alpha=$dirichlet_alpha  data_dist.max_per_client_data_size=-1 data_dist.dist_type=$dist_type dry_run=False | tee -a logs/train_pathological_text_classification_exp.log

# num_clients=7
# clients_per_round=7
# dist_type="sharded-non-iid-2"

# assert_equal $num_clients $clients_per_round
# python -m tracefl.main --multirun do_training=True exp_key=$key_start model.name=$text_models dataset.name=$text_dataset num_clients=$num_clients num_rounds=$num_rounds clients_per_round=$clients_per_round dirichlet_alpha=$dirichlet_alpha  data_dist.max_per_client_data_size=-1 data_dist.dist_type=$dist_type dry_run=False | tee -a logs/train_pathological_text_classification_exp.log
