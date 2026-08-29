from __future__ import annotations
import argparse, torch
from tokenizer import ByteTokenizer
from model_gpt import GPT


def main():
    p = argparse.ArgumentParser()