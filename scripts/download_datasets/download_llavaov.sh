# Download llava-ov
python download_llava_onevision.py --output_dir ./LLaVA-OneVision-Data

# Preprocess (Parquet->Jsonl+extract images)
python preprocess_llava_onevision.py --dataset_path ./LLaVA-OneVision-Data-processed --save_path ./LLaVA-OneVision-Data-processed

# Merge (Merge into one single training file)
python merge_llava_onevision.py --save_path ./LLaVA-OneVision-Data-processed