#!/bin/bash

## SGD with noise-level 0.35 
# python deep_hess.py --device_id=7 --sigma=0.35  --opt="SGD" --ino=1
# python deep_hess.py --device_id=7 --sigma=0.35  --opt="SGD" --ino=2

# Repeat with 0.45 sigma 

## Do for al images 1-20

# devs=(0 2 4 6 7)
devs=(0 2 4 6 7)

for sigma in 0.35 0.45;
do
    for ino in {0..19};
    do 
        # for opt in "${strings[@]}"; 
        i_dev=$(( $ino % ${#devs[@]} ))
        dev_id=${devs[$i_dev]}

        # for opt in "SGD" "SAM";
        for opt in "SGD";
        do
            python optim_expr_deep_hess.py --device_id=$dev_id \
                                            --ino=$ino \
                                            --optim=$opt \
                                            --reg=0.1 \
                                            --sigma=$sigma &

            done
        done
    done
        

## Do with reg=0.1

# Define the command to be executed
# COMMAND="python deep_hess.py --device_id "

# # Define the parameter values
# PARAMETERS=("value1" "value2" "value3")

# # Define the maximum number of parallel processes
# MAX_PARALLEL=3

# # Function to run a command with a given parameter
# run_command() {
#   local parameter="$1"
#   echo "Running command with parameter: $parameter"
#   $COMMAND "$parameter"
# }

# # Loop through the parameter values
# for parameter in "${PARAMETERS[@]}"; do
#   # Check the number of running processes
#   while [[ $(jobs -p | wc -l) -ge $MAX_PARALLEL ]]; do
#     sleep 1
#   done

#   # Run the command in the background
#   run_command "$parameter" &
# done

# # Wait for all background processes to finish
# wait
