Repository Layout (part 2)

part_2/
    orchestrator.py     # runs quick smoke-train+eval +sample
    tokenizer.py        # 2.1 byte-level tokenizer (0..255)
    dataset.py          # 2.2 dataset + batching + shift
    utils.py            # 2.3 sampling helpers (top-k/top-p)
    model_gpt.py        # tiny GPT: tok/pos emb + blocks + head
    train.py            # 2.3/2.4 training loop w/ val eval & ckpt
    sample.py           # 2.5 text generation from a checkpoint
    eval_loss.py        # 2.6 evaluate loss on a file/ckpt
    tests/
        test_tokenizer.py       #round-trip enmcode/decode
        test_dataset_shift.py   #label shift sanity
    runbs/              # created at runtime - checkpoitns & logs


Note on imports
-------
All imports are LOCAL. Run from inside part_2/
Example quick start: CPU ok
    cd part_2
    python train.py --data tiny.txt --steps 300 --sample_every 100
    python sample.py --ckpt runs/min-gpt/model_best.pt --tokens 200 --prompt ''Once upon a time'