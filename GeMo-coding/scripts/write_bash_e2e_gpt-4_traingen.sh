set -x
python write_bash_gpt4_traingen.py --input_id $1
python write_bash_gpt4_traingen_all.py --input_id $1
