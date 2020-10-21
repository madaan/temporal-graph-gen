"""Generates temporal graphs for given documents.

Usage:
    generate.py [options]

Options:
    --model-path=<str>          Path to the checkpoint
    --input-path=<str>          Path to the input file
    --output-path=<str>         Path to the output file
    --batch-size=<int>          The batch size [default: 32]
    --generation-size=<str>     What kind of generation needs to happen (large or small) [default: large]
    --use-finetuned=<str>       Whether to use finetuned model or not [default: True]

"""

from typing import List
from docopt import docopt
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)
import json
import logging
import torch
from tqdm import tqdm
import sys
sys.path.insert(0, '..')
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)
MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer)
}

GENERATION_TYPE_SMALL, GENERATION_TYPE_LARGE = "small", "large"


class Gpt2Generator(object):

    MAX_PROMPT_LEN = 440 - 5
    PAD_TOKEN = '<|pad|>'
    STOP_TOKEN = '<|endoftext|>'

    def __init__(self, model_path, generation_type, use_finetuned=True):
        self.model_path = model_path
        self.batch_size = int(args["--batch-size"])

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        self.MAX_LEN = {
            GENERATION_TYPE_SMALL: 20,
            GENERATION_TYPE_LARGE: 500
        }[generation_type]
        logger.info(f"Using {generation_type} for decoding, MAX_LEN={self.MAX_LEN}")
        if use_finetuned:
            logger.info("Using a finetuned model")
            self.config = GPT2Config.from_pretrained(self.model_path)
            model = GPT2LMHeadModel.from_pretrained(self.model_path)
            with open(f"{self.model_path}/special_tokens_map.json", "r") as f:
                special_tokens = json.load(f)
            self.tokenizer.add_special_tokens(special_tokens)
        else:
            logger.info("NOT using a finetuned model")
            model = GPT2LMHeadModel(
                config=GPT2Config.from_pretrained(pretrained_model_name_or_path=self.model_path))
        self.model = model.cuda()
        self.model.eval()

    def get_predictions_document(self, inpath, outpath):
        test_input = self.read_jsonl_to_list(inpath)
        with open(f"{outpath}", 'w') as out_file:
            for item in tqdm(test_input):
                try:
                    item["temporal_graph"] = self.get_predictions_sentence(
                        item["text"]).strip()
                    json.dump(item, out_file)
                    out_file.write('\n')
                except Exception as e:
                    logger.info(e)
                    continue

    def read_jsonl_to_list(self, pth: str) -> List[dict]:
        res = []
        with open(pth, 'r') as open_file:
            for line in open_file:
                res.append(json.loads(line))
        return res

    def get_predictions_sentence(self, input_sentence: str, beam_search: bool = False) -> str:
        encoded_prompt = self.tokenizer.encode(
            input_sentence, add_special_tokens=True, return_tensors='pt')

        if encoded_prompt.shape[1] > Gpt2Generator.MAX_PROMPT_LEN:
            encoded_prompt = encoded_prompt[:, :Gpt2Generator.MAX_PROMPT_LEN]

        encoded_prompt = encoded_prompt.cuda()
        prompt_length = encoded_prompt.shape[1]
        generation = ""
        with torch.no_grad():

            if beam_search:
                out = self.model.generate(
                    # input_ids: `torch.LongTensor` of shape `(batch_size, sequence_length)`
                    input_ids=encoded_prompt,
                    max_length=self.MAX_LEN + prompt_length,
                    num_beams=5
                )
            else:
                out = self.model.generate(
                    # input_ids: `torch.LongTensor` of shape `(batch_size, sequence_length)`
                    input_ids=encoded_prompt,
                    max_length=self.MAX_LEN + encoded_prompt.size(-1),
                    temperature=1.0,
                    top_k=0,
                    top_p=0.9,
                    do_sample=True,
                    num_return_sequences=1,
                )
            for out_seq in out:
                out_seq = out_seq[prompt_length:]
                text = self.tokenizer.decode(
                    out_seq, clean_up_tokenization_spaces=True)
                text = text[: text.find(
                    Gpt2Generator.STOP_TOKEN) if Gpt2Generator.STOP_TOKEN else None]

                generation += text
            return generation

if __name__ == '__main__':
    args = docopt(__doc__)
    generator = Gpt2Generator(
        model_path=args["--model-path"], generation_type=args["--generation-size"], use_finetuned=args["--use-finetuned"] == "True")
    generator.get_predictions_document(
        inpath=args["--input-path"],
        outpath=args["--output-path"])
