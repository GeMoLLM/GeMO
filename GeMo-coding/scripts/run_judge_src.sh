#!/bin/bash

FILEID=$1
PID=$2
PROBNAME=$3
INPUTID=$4

# Set the paths
python_file=${FILEID}_${PID}_solutions_${INPUTID}.txt

echo $python_file
stats_file="run_judge_stats/src_stats_${FILEID}_${PID}_${INPUTID}.txt"

# Clear the stats file
> "$stats_file"

# Set a timeout limit in seconds
timeout_limit=60

# Loop through test cases
for i in {0..9}; do
    input_file="test_cases/${PROBNAME}_input_$i.txt"
    expected_output_file="test_cases/${PROBNAME}_output_$i.txt"
    output_file="run_output/src_${PROBNAME}_${INPUTID}_output_$i.txt"

    # Run the Python file with the input file and capture the output and resource usage statistics
    output=$(timeout "$timeout_limit" /usr/bin/time python "$python_file" < "$input_file" 2>&1)

    # Check if the command timed out
    if [ $? -eq 124 ]; then
        echo "Test case $i timed out (Runtime: >${timeout_limit}s, Memory: N/A)" >> "$stats_file"
        continue
    fi

    # Extract runtime (elapsed time)
    runtime=$(echo "$output" | grep -oP '\d+:\d+\.\d+elapsed' | grep -oP '\d+:\d+\.\d+')

    # Extract memory usage (maxresident)
    memory_usage=$(echo "$output" | grep -oP '\d+maxresident' | grep -oP '\d+')

    # Convert runtime to milliseconds
    IFS=':' read -r minutes seconds <<< "$runtime"
    seconds=$(echo "$seconds" | sed 's/^0*//') # Remove leading zeros
    milliseconds=$(echo "($minutes * 60 + $seconds) * 1000" | bc)
    
    # Write the output to the output file
    python "$python_file" < "$input_file" > "$output_file"

    # Compare the output with the expected output
    diff -w -B "$output_file" "$expected_output_file"
    if [ $? -eq 0 ]; then
        echo "Test case $i passed (Runtime: $milliseconds ms, Memory: $memory_usage kB)" >> "$stats_file"
    else
        echo "Test case $i failed (Runtime: $milliseconds ms, Memory: $memory_usage kB)" >> "$stats_file"
    fi
done
