#!/bin/bash

current_dir=$( dirname -- "$0"; )
echo $current_dir

# COMMAND
# al_pinn.sh {tests to run} {methods to run} {repeats} {other args for python script}

# EXAMPLE COMMAND
# al_pinn.sh "0" "0 3 6" "0 1 2 3 4" "--pdebench_dir /home/a/apivich/pdebench"

pdes=(
    "--hidden_layers 8 --eqn conv-1d --use_pdebench --data_seed 40 --const 1.0 --train_steps 200000 --num_points 200 --mem_pts_total_budget 1000"
    "--hidden_layers 4 --eqn burgers-1d --use_pdebench --data_seed 20 --const 0.02 --train_steps 200000 --num_points 100 --mem_pts_total_budget 300"
)

algs=(
    "--method sampling --eig_weight_method alignment --eig_sampling pseudo --eig_memory --auto_al"
    "--method sampling --optim multiadam --eig_weight_method alignment --eig_sampling pseudo --eig_memory --auto_al"
    "--method sampling --eig_weight_method alignment --eig_sampling pseudo --eig_memory --auto_al --autoscale_loss_w_bcs --random_points_for_weights --autoscale_first"
    "--method sampling --eig_weight_method alignment --eig_sampling pseudo --eig_memory --auto_al --autoscale_loss_w_bcs"
    "--method kmeans --eig_weight_method alignment --eig_sampling pseudo --eig_memory --auto_al"
    "--method kmeans --optim multiadam --eig_weight_method alignment --eig_sampling pseudo --eig_memory --auto_al"
    "--method kmeans --eig_weight_method alignment --eig_sampling pseudo --eig_memory --auto_al --autoscale_loss_w_bcs --random_points_for_weights --autoscale_first"
    "--method kmeans --eig_weight_method alignment --eig_sampling pseudo --eig_memory --auto_al --autoscale_loss_w_bcs"
)

losses=(
    "--loss_w_bcs 1.0"
)

for j in $3; do

    for k in $1; do

        pde="${pdes[$k]}"
        echo "PDE params: $pde"

        for loss in "${losses[@]}"; do

            for m in $2; do

                alg="${algs[$m]}"

                pdeargs="$pde $alg $loss $4"

                python ${current_dir}/al_pinn.py $pdeargs
                
            done

        done

    done

done