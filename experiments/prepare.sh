SCRIPT_DIR=$(dirname $(readlink -f "$0"))
PROJETC_ROOT=$SCRIPT_DIR/..

cd $PROJETC_ROOT

python tools/preprocess_data.py \
       --input ./experiments/corpus_data/mixed_corpus.jsonl \
       --output-prefix ./experiments/corpus_data/megatron_ready \
       --tokenizer-model $SCRIPT_DIR/Qwen3-1.7B-FP8 \
       --tokenizer-type HuggingFaceTokenizer \
       --append-eod \
       --workers 16