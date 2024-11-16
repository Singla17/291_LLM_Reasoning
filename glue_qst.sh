python glue_qst_opt.py --model_checkpoint facebook/opt-1.3b  --batch_size 16 >> opt_16.txt
python glue_qst_llama.py --model_checkpoint meta-llama/Llama-2-7b-hf  --batch_size 16 >> llama_16.txt