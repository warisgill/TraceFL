

# DP-(noise0.0001+clip15)-DP-image-densenet121-mnist-faulty_clients[[]]-noise_rateNone-TClients100-fedavg-(R15-clientsPerR10)-non_iid_dirichlet0.2-batch32-epochs2-lr0.001
# DP-(noise0.0001+clip15)-DP-image-densenet121-pathmnist-faulty_clients[[]]-noise_rateNone-TClients100-fedavg-(R15-clientsPerR10)-non_iid_dirichlet0.2-batch32-epochs2-lr0.001



## ===========================================    Job 2: Provenance Analysis Differential Privacy    ===========================================
check_prov_cache=True

echo "      ****************** Differential Privacy Provenance ******************"
num_rounds=15
dp_noise="0.0001,0.0003,0.0007,0.0009,0.001,0.003" 
dp_clip="15,50"
num_clients=100
clients_per_round=10




# densenet on mnist
python -m tracefl.main --multirun check_prov_cache=$check_prov_cache parallel_processes=$parallel_processes do_provenance=True exp_key=DP-image model.name="densenet121" dataset.name="mnist" num_clients=$num_clients num_rounds=$num_rounds clients_per_round=$clients_per_round dirichlet_alpha=0.2 noise_multiplier=$dp_noise clipping_norm=$dp_clip dry_run=False | tee -a logs/prov_dp_exp_image.log 

python -m tracefl.main --multirun check_prov_cache=$check_prov_cache parallel_processes=$parallel_processes do_provenance=True exp_key=DP-image model.name="densenet121" dataset.name="pathmnist" num_clients=$num_clients num_rounds=$num_rounds clients_per_round=$clients_per_round dirichlet_alpha=0.2 noise_multiplier=$dp_noise clipping_norm=$dp_clip dry_run=False | tee -a logs/prov_dp_exp_image.log