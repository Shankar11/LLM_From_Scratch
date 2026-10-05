from __future__ import annotations
import argparse, torch
from pathlib import Path

from policy import PolicyWithValue #we will ignore the value head
from rollout import RLHFTokenizer, format_prompt_only, sample_prompts, model_logprobs
#Reward model from Part 7
import sys
from pathlib import Path as _P
sys.path.append(str(_P(__file__).parents[1]/'part_7'))
from model_reward import RewardModel #noqa: E402
from grpo_loss import ppo_policy_only_losses

@torch.no_grad()
def compute_reward(reward_model: RewardModel, tok: RLHFTokenizer, prompt_text: str, response_ids: list[int], device) -> float:
    #Build full formatted text (as in your PPO)
    from part_6.formatters import Example, format_example
    resp_text = tok.decode(response_ids)
    text = format_example(Example(prompt_text, resp_text))
    ids = tok.encode(text)
    x = torch.tensor([ids[:tok.block_size]], dtype=torch.long, device=device)
    r = reward_model(x)
    return float(r[0].item())

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=str, default='runs/grpo-demo')
    p.add_argument('--policy_ckpt', type = str, required=True, help='SFT checkpoint (Part 6)')
    p.add_argument('--reward_ckpt', type=str, required=True, help='Reward model checkpoint (Part 7)')
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--batch_prompts', type=int, default=32, help='number of distinct prompts per step (before grouping)')
    p.add_argument('--group_size', type=int, default=4, help='completions per prompt')
    p.add_argument('--block_size', type=int, default=256)
    p.add_argument('--resp_len', type=int, default=64)
    p.add_argument('--kl_coef', type = float, default=0.01)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--bpe_dir', type=str, default=None)
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    #tokenizer
    tok = RLHFTokenizer(block_size=args.block_size, bpe_dir=args.bpe_dir)

    #load SFT policy (and a frozen reference)
    ckpt = torch.load(args.policy_ckpt, map_location = device)
    cfg = ckpt.get('config', {})
    vocab_size = cfg.get('vocab_size', tok.vocab_size)
    block_size = cfg.get('block_size', tok.block_size)
    n_layer = cfg.get('n_layer',2)
    n_head = cfg.get('n_head',2)
    n_embd = cfg.get('n_embd', 128)

    policy = PolicyWithValue(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    policy.lm.load_state_dict(ckpt['model'])
    policy.eval()

    ref = PolicyWithValue(vocab_size,block_size, n_layer,n_head, n_embd).to(device)
    ref.lm.load_state_dict(ckpt['model'])
    for p_ in ref.parameters():
        p_.requires_grad_(False)
    ref.eval()

    #Reward MOdel
    rckpt = torch.load(args.reward_ckpt, map_location=device)
    rm = RewardModel(vocab_size=rckpt['config'].get('vocab_size',tok.vocab_size),
                     block_size=rckpt['config'].get('block_size', tok.block_size),
                     n_layer = rckpt['config'].get('n_layer', 4),
                     n_head = rckpt['config'].get('n_head',4),
                     n_embd = rckpt['config'].get('n_embd', 256)).to(device)
    rm.load_state_dict(rckpt['model'])
    rm.eval()

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.9, 0.999))

    #small prompt pool (reuse your helper)
    prompts_pool = sample_prompts(16)

    step=0
    pool_idx = 0
    G = args.group_size

    while
    