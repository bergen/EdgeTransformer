import sys
import time
import argparse
from dataclasses import dataclass, fields
import os
import random
from typing import Optional, Tuple
import json

import transformers
from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration, 
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM, 
    PreTrainedTokenizerFast,
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)
from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers.optimization import AdamW
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.trainer_utils import get_last_checkpoint


from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers
from tokenizers.processors import TemplateProcessing


from model import EdgeConfig, EdgeTransfomerForConditionalGeneration


logger = transformers.logging.get_logger(__name__)


def get_n_params(model):
    total_n_params = 0
    for p in list(model.parameters()):
        n_params = 1
        for dim_len in list(p.size()):
            n_params *= dim_len
        total_n_params += n_params
    return total_n_params


def train_tokenizer(dataset):
    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))

    tokenizer.normalizer = normalizers.Sequence([normalizers.Lowercase()])
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.post_processor = TemplateProcessing(
        single="$A [EOS]", 
        special_tokens=[("[EOS]", 2)]
    )

    trainer = trainers.WordLevelTrainer(
        special_tokens=["[PAD]", "[UNK]", "[EOS]"]
    )

    def tokenizer_iterator(batch_size=1000):
        for i in range(0, len(dataset["train"]), batch_size):
            yield dataset["train"][i : i + batch_size]["query"]
        for i in range(0, len(dataset["train"]), batch_size):
            yield dataset["train"][i : i + batch_size]["question"]

    tokenizer.train_from_iterator(tokenizer_iterator(), trainer=trainer)
    return tokenizer


def try_to_load_tokenizer(path):
    try:
        return Tokenizer.from_file(path)
    except:
        return None


class WordTokenizer(PreTrainedTokenizerFast):

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # Mostly copied from T5PretrainedTokenizer
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        )
        with open(out_vocab_file, 'w') as dst:
            json.dump(self.vocab, dst)

        return (out_vocab_file,)


@dataclass
class CFQTrainingArguments(Seq2SeqTrainingArguments):
    arch : str = "edge"
    from_scratch : bool = True
    reuse_first_block : bool = True
    max_label_length : int = 50
    max_total_length : int = 1000
    split : str = 'mcd1'  
    filter_test : bool = False
    num_training_examples : int = 0
    seed: int = 42


if __name__ == "__main__":
    transformers.logging.set_verbosity(transformers.logging.INFO)


    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default='output_edge')
    cmd_args = parser.parse_args()

    if cmd_args.local_rank <= 0:
        if not os.path.exists(cmd_args.output_dir):
            os.mkdir(cmd_args.output_dir)
        else:
            raise ValueError("Output directory already exists")

    # create and parse configuration
    args = CFQTrainingArguments(
        output_dir=cmd_args.output_dir,
        num_train_epochs=100,
        learning_rate=0.0006,
        max_grad_norm=100,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        save_steps=1000,
        warmup_steps=1000,
        eval_steps=3000,
        logging_steps=100,        
        load_best_model_at_end=False,
        predict_with_generate=True,
        evaluation_strategy='steps',
        metric_for_best_model='acc',
        ignore_data_skip=True,
        local_rank=cmd_args.local_rank,
        fp16=True,
    )

    print(f'My rank is {cmd_args.local_rank}, my seed is {args.seed}')            

    # create and parse the model configuration
    if args.arch == 'edge':
        model_config = EdgeConfig()
    else:
        model_config = AutoConfig.from_pretrained(args.arch)
    model_config.max_length = 70

    # derive some configs settings from others
    if args.arch != 'edge':
        model_config.d_ff = 4 * model_config.d_model
        model_config.d_kv = model_config.d_model // model_config.num_heads

    if args.arch == 'edge':
        model = EdgeTransfomerForConditionalGeneration(model_config)
    elif args.from_scratch:
        model = AutoModelForSeq2SeqLM.from_config(model_config)
        if args.reuse_first_block: 
            print("WARNING: use the hacky method for sharing parameters across layers")
            def reuse_first_block(model_part):
                for i in range(1, len(model_part.block)):
                    model_part.block[i] = model_part.block[0]        
            reuse_first_block(model.encoder)
            reuse_first_block(model.decoder)
    else:
        assert not args.reuse_first_block
        model = AutoModelForSeq2SeqLM.from_pretrained(args.arch)
        model.config.max_length = model_config.max_length

    print(f'{get_n_params(model)} trainable parameters')    
    dataset = load_dataset("cfq", args.split)

    if args.from_scratch:
        tokenizer_save_path = f"{args.output_dir}/tokenizer.json"
        backend_tokenizer = try_to_load_tokenizer(tokenizer_save_path)
        if not backend_tokenizer:
            if args.local_rank <= 0:
                backend_tokenizer = train_tokenizer(dataset)
                backend_tokenizer.save(tokenizer_save_path)
                print(f"Process {args.local_rank} has built and saved the tokenizer.")
            else:
                while not backend_tokenizer:
                    print(f"Process {args.local_rank} waiting for Process 0 to produce the tokenizer")
                    time.sleep(10)
                    backend_tokenizer = try_to_load_tokenizer(tokenizer_save_path)
        else:
            print("Using the loaded tokenizer")
        tokenizer = WordTokenizer(tokenizer_object=backend_tokenizer)
        
        tokenizer.pad_token = "[PAD]"
        tokenizer.eos_token = "[EOS]"
        model.config.decoder_start_token_id = tokenizer.vocab['[PAD]']
        model.config.pad_token_id = tokenizer.vocab['[PAD]']
        model.config.eos_token_id = tokenizer.vocab['[EOS]']
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.arch)

    def tokenize(entry):
        tokenized_question = tokenizer(entry['question'], add_special_tokens=True)
        tokenized_query = tokenizer(entry['query'], add_special_tokens=True)
        # note: no decoder_attention_mask because that labels will be 
        # padded with -100 and not taken into account for the loss
        return {
            'input_ids': tokenized_question['input_ids'],
            'attention_mask': tokenized_question['attention_mask'],
            'labels': tokenized_query['input_ids'],
        }
    dataset = dataset.map(tokenize, batched=True)
    def is_manageable_length(ex):
        word_tokenized_label_length = len(ex['query'].split())
        word_tokenized_question_length = len(ex['question'].split())
        return (word_tokenized_label_length < args.max_label_length 
                and word_tokenized_question_length + word_tokenized_label_length < args.max_total_length)
    dataset['train'] = dataset['train'].filter(is_manageable_length)
    print(f"{len(dataset['train'])} training examples left after filtering the dataset")
    if args.filter_test:
        dataset['test'] = dataset['test'].filter(is_manageable_length)
        print(f"{len(dataset['test'])} test examples left after filtering the dataset")

    if args.split.startswith('mcd'):
        n = len(dataset['train'])        
        validation_indices = set(random.Random(args.seed).sample(list(range(n)), 1000))
        print('drawing validation indices from the training set')
        training_indices = [idx for idx in list(range(n))
                            if idx not in validation_indices]
        original_train = dataset['train']
        dataset['train'] = original_train.select(training_indices)
        dataset['valid'] = original_train.select(validation_indices)
    elif args.split.startswith('random_split'):
        n = len(dataset['test'])
        validation_indices = set(random.Random(args.seed).sample(list(range(n)), 1000))
        print('drawing validation indices from the test set')
        test_indices = [idx for idx in list(range(n))
                        if idx not in validation_indices]
        original_test = dataset['test']
        dataset['test'] = original_test.select(test_indices)
        dataset['valid'] = original_test.select(validation_indices)
    else:
        raise ValueError()

    if args.num_training_examples:
        training_indices = set(
            random.Random(args.seed).sample(list(range(len(dataset['train']))), args.num_training_examples)
        )
        dataset['train'] = dataset['train'].select(training_indices)
        print(f"Keep {len(dataset['train'])} examples in the training set")


    data_collator = DataCollatorForSeq2Seq(tokenizer)
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        n_acc = 0
        for pred, label in zip(preds, labels):
            try:
                pred = list(pred)
                label = list(label)
                pred_trimmed = pred[:pred.index(model.config.eos_token_id)]
                label_trimmed = label[:label.index(model.config.eos_token_id)]
                if list(pred_trimmed[1:]) == list(label_trimmed):
                    n_acc += 1
            except ValueError:
                pass
        return {'acc': n_acc / preds.shape[0]}
    # note: the default optimizer is AdamW        
    trainer = Seq2SeqTrainer(
        args=args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['valid'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        model=model,
        compute_metrics=compute_metrics)
    trainer.train(resume_from_checkpoint=get_last_checkpoint(args.output_dir))
    trainer.evaluate(dataset['test'], metric_key_prefix='test')
