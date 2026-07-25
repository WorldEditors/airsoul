#! /bin/bash
#
if [[ $# -lt 1 ]]; then
	echo "Usage: $0 output_directory [task_source_file]"
exit 1
fi
echo "Output to $1"

task_args=(--task_source NEW)
if [[ $# -ge 2 ]]; then
  task_args=(--task_source FILE --task_file "$2")
fi

python3 gen_maze_record.py \
  --output_path "$1" \
  "${task_args[@]}" \
  --max_steps 4000 \
  --n_range 9,21 \
  --epochs 1 \
  --workers 4
