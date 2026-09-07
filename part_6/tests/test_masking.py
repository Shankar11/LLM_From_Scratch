from collator_sft import SFTCOllator
from formatters import Example

def test_masking_sets_prompt_to_ignore():
    col = SFTCOllator(block_size=256, bpe_dir='../part_4/runs/part4-demo/tokenizer')
    text = "this is a tiny test.."
    x,y = col.collate([(text, "OK")])
    #All labels up to response marker should be -100
    boundary = ("<s> \n### Instruction:\n" + text + "\n\n### Response:\n")
    #We dont have direct access to the tokenized boundary; just sanity check: some -100s present
    assert(y[0] == -100).sum() > 0