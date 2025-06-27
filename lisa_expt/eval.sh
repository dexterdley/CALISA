CUDA_VISIBLE_DEVICES=0 python ./lisa_expt/eval.py --version='xinlai/LISA-13B-llama2-v1' --workers=0 --noise=0.750 --importance=True
CUDA_VISIBLE_DEVICES=0 python ./lisa_expt/eval.py --version='xinlai/LISA-13B-llama2-v1' --workers=0 --noise=0.750 --importance=False
CUDA_VISIBLE_DEVICES=7 python ./lisa_expt/eval_plus.py --version='Senqiao/LISA_Plus_7b' --workers=0 --noise=0.750 --importance=
#  CUDA_VISIBLE_DEVICES=0 python ./lisa_expt/eval.py --version='xinlai/LISA-7B-v1' --workers=0 --noise=0.75 --importance=True