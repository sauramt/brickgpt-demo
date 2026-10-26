import time

from brickgpt.data import BrickStructure
from brickgpt.models import LLM, BrickGPT, BrickGPTConfig, create_instruction

BRICKGPT_PATH = 'AvaLovelace/BrickGPT'


def test_llm():
    """
    Tests the LLM model by generating two different continuations from a prompt.
    """
    llm = LLM('meta-llama/Llama-3.2-1B-Instruct')
    prompt = 'A fun fact about llamas is:'
    output = llm(prompt, max_new_tokens=10)

    # First continuation
    llm.save_state()
    output_continuation = llm(max_new_tokens=10)
    print(prompt + '|' + output + '|' + output_continuation)

    # Second continuation
    llm.rollback_to_saved_state()
    output_continuation = llm(max_new_tokens=10)
    print(prompt + '|' + output + '|' + output_continuation)


def test_finetuned_llm():
    """
    Tests running the finetuned BrickGPT model with no other guidance (e.g. rejection sampling).
    """
    llm = LLM(BRICKGPT_PATH)
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': create_instruction('A basic chair with four legs.')},
    ]
    prompt = llm.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt')

    prompt_txt = llm.tokenizer.decode(prompt[0])
    print(prompt_txt)
    start_time = time.time()
    output = llm(prompt, max_new_tokens=8192)
    end_time = time.time()
    print(output)
    print(f'Time taken: {end_time - start_time:.2f}s')


def test_infer():
    """
    Runs BrickGPT inference on a simple prompt.
    """
    brickgpt = BrickGPT(BrickGPTConfig(BRICKGPT_PATH))

    start_time = time.time()
    output = brickgpt('A basic chair with four legs.')
    end_time = time.time()

    print(output['bricks'])
    print('# of bricks:', len(output['bricks']))
    print('Brick rejection reasons:', output['rejection_reasons'])
    print('# regenerations:', output['n_regenerations'])
    print(f'Time taken: {end_time - start_time:.2f}s')


def test_finish_partial_structure():
    partial_structure_txt = '1x1 (2,19,0)\n1x4 (2,15,0)\n1x8 (2,7,0)\n1x1 (1,6,0)\n2x2 (0,18,0)\n2x1 (0,17,0)\n2x6 (0,11,0)\n'
    partial_bricks = BrickStructure.from_txt(partial_structure_txt)
    brickgpt = BrickGPT(BrickGPTConfig(BRICKGPT_PATH, max_bricks=1, max_regenerations=0))
    bricks, rejections = brickgpt._generate_structure(
        'An elongated, rectangular vessel with layered construction, central recess, and uniform edges.', partial_bricks)

    print(bricks)
    print('# of bricks:', len(bricks))
    print('Brick rejection reasons:', rejections)
