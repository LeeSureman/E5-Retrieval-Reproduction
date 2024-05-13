import copy, os, random, json, math, pathlib, jsonlines
import torch
from torch.utils.data import Dataset
import transformers
from retriever_trainer import Retriever_Trainer
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model
import torch.distributed as dist
from torch.utils.data import Dataset
from modeling_retriever import DenseEmbedder
import task_config
from args_config import ModelArguments, DataArguments, TrainingArguments, LoraArguments

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def trainer_save_model_safe(trainer: transformers.Trainer):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(trainer.model, StateDictType.FULL_STATE_DICT, save_policy):
        trainer.save_model()


def tokenize_into_ids_with_special_token_for_sequence_embedding(input_str,
                                                                tokenizer, 
                                                                representation_token,
                                                                representation_token_num, 
                                                                max_length):
    representation_special_tokens = representation_token * representation_token_num
    representation_special_ids = tokenizer(representation_special_tokens)['input_ids'][1:]
    representation_special_tokens_num = len(representation_special_ids)

    input_str_encoded = tokenizer(
        input_str,
        max_length=max_length - representation_special_tokens_num,
        return_token_type_ids=False,
        return_attention_mask=False,
        padding=False,
        truncation=True
    )

    input_str_encoded['input_ids'] = input_str_encoded['input_ids'] + representation_special_ids

    result = tokenizer.pad(
        input_str_encoded,
        padding="max_length",
        pad_to_multiple_of=8,
        return_attention_mask=True,
        return_tensors=None,
        max_length=max_length
    )
    return result


def load_retrieval_raw_data(root_dir):
    '''Use filename as task name'''
    result_data_dict = {}
    data_file_name_s = os.listdir(root_dir)
    for data_file_name in data_file_name_s:
        data_fp = os.path.join(root_dir, data_file_name)
        data_task_name = data_file_name.split('.')[0]
        tmp_task_data = list(jsonlines.open(data_fp))
        for tmp_js in tmp_task_data:
            tmp_js['task'] = data_task_name  # Add new field named `task`
        result_data_dict[data_task_name] = tmp_task_data
    return result_data_dict


class RetrievalDataset(Dataset):
    def __init__(
        self, 
        args_dict, 
        data_root_dir, 
        n_hard_negative, 
        tokenizer,
        representation_id,
        representation_token_num, 
        query_max_length, 
        doc_max_length, 
        wrap_q_p
    ):
        
        self.args_dict = args_dict
        self.batch_same_task = args_dict['training_args']['batch_same_task']
        self.tokenizer = tokenizer
        self.raw_data_task_to_dict = load_retrieval_raw_data(data_root_dir)
        self.raw_data = None
        self.n_hard_negative = n_hard_negative
        assert tokenizer.padding_side == 'right', '`padding_side` must be set to right'
        self.representation_id = representation_id
        rank0_print('representation_id: {}'.format(representation_id))
        self.representation_token = self.tokenizer.convert_ids_to_tokens(representation_id)
        self.representation_token_num = representation_token_num
        self.query_max_length = query_max_length
        self.doc_max_length = doc_max_length
        self.wrap_q_p = wrap_q_p

        task_weight_dict = task_config.task_to_sample_weight
        for k, v in task_weight_dict.items():
            rank0_print('{}: {}'.format(k, v))

        self.update_one_epoch_data()
        self.task_to_query_type = task_config.task_to_query_type
        self.task_to_passage_type = task_config.task_to_passage_type
        self.instruction_template = 'Given {}, retrieve {}.'

        for k in self.raw_data_task_to_dict:
            if k != 'synthetic':
                rank0_print('instruction of task {}: {}'.format(k, self.instruction_template.format(self.task_to_query_type[k],
                                                                                                    self.task_to_passage_type[k])))

        if 'synthetic' in self.raw_data_task_to_dict:
            rank0_print('synthetic data instruction examples:')
            for i in range(3):
                rank0_print(self.raw_data_task_to_dict['synthetic'][i]['instruction'])

        self.task_to_tid = {}
        task_weight_dict = task_config.task_to_sample_weight
        tasks = sorted(task_weight_dict.keys())

        self.tid_to_task = tasks
        for i, t in enumerate(tasks):
            self.task_to_tid[t] = i

        for i, t in enumerate(tasks):
            rank0_print('task {}: {}'.format(i, t))

    def update_one_epoch_data(self, epoch=-1):
        rank0_print('re-sample this epoch data')
        if self.batch_same_task:
            total_batch = self.args_dict['training_args']['per_device_train_batch_size'] * dist.get_world_size()
            rank0_print('total batch: {}'.format(total_batch))
            this_epoch_data = []
            this_epoch_data_batch = []
            task_to_data_batch = {}

            task_weight_dict = task_config.task_to_sample_weight
            for k in self.raw_data_task_to_dict:
                assert k in task_weight_dict, '{} not in {}'.format(k, self.args_dict['data_args']['data_sample_config'])

            for task, task_data in self.raw_data_task_to_dict.items():
                task_weight = task_weight_dict[task]
                sampled_data_data = random.sample(task_data, int(task_weight * len(task_data)))
                task_to_data_batch[task] = []
                for i in range(len(sampled_data_data) // total_batch):
                    task_to_data_batch[task].append(sampled_data_data[i * total_batch: (i + 1) * total_batch])
                task_to_data_batch[task] = list(filter(lambda x:len(x)==total_batch, task_to_data_batch[task]))

            for task, task_batchs in task_to_data_batch.items():
                rank0_print('{} training size: {}'.format(task, len(task_batchs) * total_batch))
                random.shuffle(task_batchs)
                this_epoch_data_batch.extend(task_batchs)

            random.shuffle(this_epoch_data_batch)
            for batchs in this_epoch_data_batch:
                this_epoch_data.extend(batchs)

        else:
            this_epoch_data = []
            task_weight_dict = task_config.task_to_sample_weight
            for k in self.raw_data_task_to_dict:
                assert k in task_weight_dict, '{} not in {}'.format(k, self.args_dict['data_args']['data_sample_config'])

            for task, task_data in self.raw_data_task_to_dict.items():
                task_weight = task_weight_dict[task]
                sampled_data_data = random.sample(task_data, int(task_weight * len(task_data)))
                this_epoch_data.extend(sampled_data_data)
                rank0_print('{}: {}'.format(task, len(sampled_data_data)))

            rank0_print('re-sample this epoch data finish, total:{}'.format(len(this_epoch_data)))
            random.Random(1208 + epoch).shuffle(this_epoch_data)

        self.raw_data = this_epoch_data
        rank0_print('self.raw_data:{}'.format(len(self.raw_data)))

    def __len__(self):
        return len(self.raw_data)

    def wrap_query(self, query, task=None, instruction=None):
        if self.wrap_q_p == 'query_or_passage':
            result = 'query: {}'.format(query)
        elif self.wrap_q_p == 'instruction':
            assert task is not None
            if task != 'synthetic':
                assert instruction is None
                tmp_instruction = self.instruction_template.format(self.task_to_query_type[task], self.task_to_passage_type[task])
            else:
                assert instruction is not None
                tmp_instruction = instruction.strip()
            result = '{}\n{}'.format(tmp_instruction, query)
        else:
            raise NotImplementedError
        return result

    def wrap_passage(self, passage, task=None):
        passage_split = passage.split('\n\n')
        title = passage_split[0]
        content = '\n\n'.join(passage_split[1:])
        passage = '{} {}'.format(title, content).strip()
        if self.wrap_q_p == 'query_or_passage':
            result = 'passage: {}'.format(passage)
        elif self.wrap_q_p == 'instruction':
            result = passage
        else:
            raise NotImplementedError
        return result

    def __getitem__(self, i):
        raw_data = self.raw_data[i]
        tid = self.task_to_tid[raw_data['task']]
        task_name = raw_data['task']
        query = raw_data['query']

        if 'instruction' in raw_data:
            query = self.wrap_query(query, task=task_name, instruction=raw_data['instruction'])
        else:
            query = self.wrap_query(query, task=task_name)

        query_encoded = tokenize_into_ids_with_special_token_for_sequence_embedding(query, 
                                                                                    self.tokenizer,
                                                                                    self.representation_token,
                                                                                    self.representation_token_num,
                                                                                    self.query_max_length)
        query_ids = query_encoded['input_ids']
        query_attention_mask = query_encoded['attention_mask']

        positive_doc = raw_data['positive'][0]
        positive_doc = self.wrap_passage(positive_doc, task=task_name)
        positive_doc_encoded = tokenize_into_ids_with_special_token_for_sequence_embedding(positive_doc, 
                                                                                           self.tokenizer,
                                                                                           self.representation_token,
                                                                                           self.representation_token_num,
                                                                                           self.doc_max_length)
        positive_doc_ids = positive_doc_encoded['input_ids']
        positive_doc_attention_mask = positive_doc_encoded['attention_mask']

        if 'negative' in raw_data:
            all_hard_negative = raw_data['negative']
            if self.n_hard_negative > 0 and len(all_hard_negative) == 0:
                raise NotImplementedError
            while len(all_hard_negative) < self.n_hard_negative:
                all_hard_negative += all_hard_negative
            hard_negative_docs = random.sample(all_hard_negative, k=self.n_hard_negative)
        else:
            hard_negative_docs = []

        hard_negative_docs_ids = []
        hard_negative_docs_attention_mask = []
        for hard_negative_doc in hard_negative_docs:
            hard_negative_doc = self.wrap_passage(hard_negative_doc, task=task_name)
            hard_negative_doc_encoded = tokenize_into_ids_with_special_token_for_sequence_embedding(hard_negative_doc,
                                                                                                    self.tokenizer,
                                                                                                    self.representation_token,
                                                                                                    self.representation_token_num,
                                                                                                    self.doc_max_length)
            hard_negative_docs_ids.append(hard_negative_doc_encoded['input_ids'])
            hard_negative_docs_attention_mask.append(hard_negative_doc_encoded['attention_mask'])

        if len(hard_negative_docs_ids) > 0:
            result = dict(
                query_ids=query_ids, 
                query_attention_mask=query_attention_mask,
                positive_doc_ids=positive_doc_ids,
                positive_doc_attention_mask=positive_doc_attention_mask,
                hard_negative_docs_ids=hard_negative_docs_ids,
                hard_negative_docs_attention_mask=hard_negative_docs_attention_mask,
                task_id=tid
            )
        else:
            result = dict(
                query_ids=query_ids, 
                query_attention_mask=query_attention_mask,
                positive_doc_ids=positive_doc_ids, 
                positive_doc_attention_mask=positive_doc_attention_mask,
                task_id=tid
            )
        return result


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, LoraArguments))
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    if training_args.per_device_train_batch_size <= training_args.chunk_sizes:
        print('per gpu batch size < grad_cache chunk size, so disable grad_cache')
        training_args.do_grad_cache = False

    assert training_args.batch_same_task is not None, '`batch_same_task` should be specified'

    args_dict = {}
    args_dict['model_args'] = model_args.__dict__
    args_dict['data_args'] = data_args.__dict__
    args_dict['training_args'] = training_args.__dict__
    args_dict['lora_args'] = lora_args.__dict__

    args_output_fp = '{}/args_dict.jsonl'.format(training_args.output_dir)
    with jsonlines.open(args_output_fp, 'w') as args_output_f:
        args_dict_to_output = copy.deepcopy(args_dict)
        for _, v1 in args_dict_to_output.items():
            for k2, v2 in v1.items():
                try:
                    json.dumps(v2)
                except:
                    v1[k2] = None

        args_output_f.write(args_dict_to_output)

    global local_rank
    local_rank = training_args.local_rank
    rank0_print('args_dict has been saved to {}'.format(os.path.abspath(args_output_fp)))

    rank0_print('*' * 30)
    rank0_print('model_args:')
    rank0_print('*' * 30)
    for k, v in model_args.__dict__.items():
        rank0_print('{}: {}'.format(k, v))

    rank0_print('*' * 30)
    rank0_print('data_args:')
    rank0_print('*' * 30)
    for k, v in data_args.__dict__.items():
        rank0_print('{}: {}'.format(k, v))

    rank0_print('*' * 30)
    rank0_print('training_args:')
    rank0_print('*' * 30)
    for k, v in training_args.__dict__.items():
        rank0_print('{}: {}'.format(k, v))

    rank0_print('*' * 30)
    rank0_print('lora_args:')
    rank0_print('*' * 30)
    for k, v in lora_args.__dict__.items():
        rank0_print('{}: {}'.format(k, v))

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    config.use_cache = False
    config.output_hidden_states = True

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation="flash_attention_2" if training_args.flash_attention else None,
        torch_dtype=torch.bfloat16 if training_args.bf16 else None
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        use_fast=False,
        trust_remote_code=model_args.trust_remote_code,
    )
   
    representation_id = model_args.representation_id
    if tokenizer.pad_token != tokenizer.unk_token:
        tokenizer.pad_token = tokenizer.unk_token

    if lora_args.use_lora:
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()
        
        model.print_trainable_parameters()

    if hasattr(config, 'attention_dropout'):
        rank0_print('model.attention_dropout: {}'.format(model.config.attention_dropout))
        
    model = DenseEmbedder(model, representation_id=representation_id, config=config)
    model.representation_token_num = model_args.representation_token_num

    # Load data
    retrieval_train_dataset = RetrievalDataset(
        args_dict, 
        data_args.data_root_dir, 
        training_args.n_hard_negative,
        tokenizer,
        representation_id, 
        model_args.representation_token_num,
        query_max_length=training_args.query_max_length,
        doc_max_length=training_args.doc_max_length,
        wrap_q_p=model_args.wrap_q_p
    )

    # Start trainner
    trainer = Retriever_Trainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        train_dataset=retrieval_train_dataset
    )

    if training_args.batch_same_task:
        rank0_print('set trainer.accelerator for batch_same_task=True')
        trainer.accelerator.device_placement = True
        trainer.accelerator.dispatch_batches = True

    trainer.set_data_collator()
    if training_args.do_grad_cache:
        trainer.chunk_sizes = training_args.chunk_sizes
    else:
        trainer.chunk_sizes = training_args.per_device_train_batch_size

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save model
    trainer.save_state()
    if trainer.is_deepspeed_enabled:
        trainer.save_model()
    else:
        trainer_save_model_safe(trainer)


if __name__ == "__main__":
    train()
