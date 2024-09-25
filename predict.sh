log_dir=log/20240617100047
python predict.py --dataPath data/uspto_condition_split -bs 512 --device 0 --log_dir ${log_dir} --best
python convert.py --log_dir ${log_dir}
python eval_ours.py --log_dir ${log_dir}