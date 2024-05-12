import torch
from transformers import Trainer
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.utils import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    PushInProgress,
    can_return_loss,
    find_labels,
    is_accelerate_available,
    is_apex_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_in_notebook,
    is_ipex_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_compile_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_tpu_available,
    logging,
    strtobool,
)

import tqdm
from collections import defaultdict

import torch.nn as nn

if is_peft_available():
    from peft import PeftModel

from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
# from modeling_retriever import retrieval_forward
from transformers.data.data_collator import DataCollatorWithPadding
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.tokenization_utils_base import BatchEncoding
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, DistributedSampler
import datasets
from transformers.utils import is_datasets_available
from transformers.trainer_utils import seed_worker
from accelerate import Accelerator, skip_first_batches

from utils import _setup_logger

from accelerate.utils import (
    DistributedDataParallelKwargs,
    DistributedType,
    GradientAccumulationPlugin,
    load_fsdp_model,
    load_fsdp_optimizer,
    save_fsdp_model,
    save_fsdp_optimizer,
)

from transformers.trainer_pt_utils import (
    AcceleratorConfig,
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_dataloader_sampler,
    get_model_param_count,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
    remove_dummy_checkpoint,
)

logger = _setup_logger()

import torch.distributed as dist

def gather_tensor(t):
    gathered = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, t)
    gathered = torch.cat(gathered, dim=0)
    return gathered


class MyDataCollatorWithPadding(DataCollatorWithPadding):
    """
    Data collator that will dynamically pad the inputs received.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # print('before MyDataCollatorWithPadding:\n{}'.format(features))

        batch_size = len(features)


        batch_outputs = {}
        for i in range(batch_size):
            inputs = features[i]
            for key, value in inputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)



        result = BatchEncoding(batch_outputs, tensor_type=self.return_tensors)

        # print('after MyDataCollatorWithPadding:\n{}'.format(result))

        return result
def rank0_print(*args):
    if dist.get_rank() == 0:
        print(*args)
from torch.nn.utils.rnn import pad_sequence
def get_collate_fn(tokenizer) -> Callable[[List[Dict]], Dict[str, Union[List, torch.Tensor]]]:
    """Returns a collate function for the given tokenizer.

       The collate function takes a list of examples (dicts, where values are lists of
         ints [tokens] or strings [the original texts]) and returns a batch of examples,
         PyTorch tensors padded to the maximum length. Strings are passed through."""

    def collate_fn(batch):
        # first, pad everything to the same length
        batch_size = len(batch)
        padded_batch = {}
        for k in batch[0].keys():
            if 'prompt' in k:
                continue
            if k.endswith('_input_ids') or k.endswith('_attention_mask') or k.endswith('_labels'):
                if 'prompt' in k:  # adapted from https://stackoverflow.com/questions/73256206
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith('_input_ids'):
                    padding_value = tokenizer.pad_token_id
                elif k.endswith('_labels'):
                    padding_value = -100
                elif k.endswith('_attention_mask'):
                    padding_value = 0
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                pad_num_to_model_max_length = tokenizer.model_max_length - padded_batch[k].size()[1]
                extra_pad_tokens = torch.full(size=[batch_size, pad_num_to_model_max_length], fill_value=padding_value)
                padded_batch[k] = torch.cat([padded_batch[k], extra_pad_tokens],dim=1)
                if 'prompt' in k:  # for the prompt, flip back so padding is on left side
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
                # rank0_print('{}: {}'.format(k, padded_batch[k].size()))
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch

    return collate_fn

def _is_peft_model(model):
    if is_peft_available():
        classes_to_check = (PeftModel,) if is_peft_available() else ()
        # Here we also check if the model is an instance of `PeftMixedModel` introduced in peft>=0.7.0: https://github.com/huggingface/transformers/pull/28321
        if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
            from peft import PeftMixedModel

            classes_to_check = (*classes_to_check, PeftMixedModel)
        return isinstance(model, classes_to_check)
    return False

from transformers import TrainerCallback

class MyCallback(TrainerCallback):

    def __init__(self, trainer=None):
        self.trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        # 在训练开始时执行的操作
        # self.trainer.eval_dpo_data()
        pass

    def on_train_end(self, args, state, control, **kwargs):
        # 在训练结束时执行的操作
        pass

    def on_epoch_begin(self, args, state, control, **kwargs):
        # 在每个epoch开始时执行的操作
        pass

    def on_epoch_end(self, args, state, control, **kwargs):
        # 在每个epoch结束时执行的操作
        self.trainer.eval_dpo_data()
        pass

    def on_step_begin(self, args, state, control, **kwargs):
        # 在每个步骤（batch）开始时执行的操作
        pass

    def on_step_end(self, args, state, control, **kwargs):
        # 在每个步骤（batch）结束时执行的操作
        # rank0_print(state.global_step)
        if state.global_step % 75 == 0:
            self.trainer.eval_dpo_data()
        pass

from modeling_dpo import DPO_Forwarder
def formatted_dict(d: Dict) -> Dict:
    """Format a dictionary for printing."""
    return {k: (f"{v:.5g}" if type(v) == float else v) for k, v in d.items()}
class DPO_Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        policy = model
        if self.dpo_args.name == 'dpo':
            reference_model = self.reference_model
        else:
            reference_model = None
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        dpo_forwarder = DPO_Forwarder(policy, reference_model, rank, world_size)

        loss, metrics = dpo_forwarder.get_batch_metrics(inputs, self.dpo_args)

        if self.dpo_args.name == 'dpo':

            mean_train_metrics = {k: sum(v) / len(v) for k, v in metrics.items()}

            rank0_print(f'train stats: {formatted_dict(mean_train_metrics)}')

        outputs = {'loss': loss}


        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if is_peft_available() and isinstance(unwrapped_model, PeftModel):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def set_data_collator(self):
        self.data_collator = get_collate_fn(self.tokenizer)

    # def training_step(self, *args, **kwargs):
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        result = super().training_step(model, inputs)
        # for debug
        # if self.args.local_rank == 0:
        #     # print(inputs.keys())
        #     print("query_ids:{}".format(inputs['query_ids'].size()))
        #     print('query_ids[:2]:{}'.format(inputs['query_ids'][:2,:7]))
        #     print('query_ids[:-2]:{}'.format(inputs['query_ids'][-2:, :7]))
        #
        #     print('hard_negative_docs_ids:{}'.format(inputs['hard_negative_docs_ids'].size()))
        #     print('hard_negative_docs_ids:{}'.format(inputs['hard_negative_docs_ids'][:2,:1,:7]))
        #
        #     # print('positive_doc_ids:{}'.format(inputs['positive_doc_ids'][:2, :7]))
        #     print('self.model.model.layers.12.self_attn.k_proj.weight.grad:\n{}'.format(self.model.model.model.layers[12].self_attn.k_proj.weight.grad.size()))
        #     print('self.model.model.layers.12.self_attn.k_proj.weight.grad:\n{}'.format(self.model.model.model.layers[12].self_attn.k_proj.weight.grad.view(1024,4096)[:3,:4]))
        #     print('self.model.model.layers.31.self_attn.k_proj.weight.grad:\n{}'.format(
        #         self.model.model.model.layers[31].self_attn.k_proj.weight.grad.size()))
        #     print('self.model.model.layers.31.self_attn.k_proj.weight.grad:\n{}'.format(self.model.model.model.layers[31].self_attn.k_proj.weight.grad.view(1024,4096)[:3,:4]))
        # exit()

        return result

    def _get_train_sampler(self):
        if self.args.sequential_sampler:
            print('use sequential sampler')
            return SequentialSampler(self.train_dataset)
        else:
            return super()._get_train_sampler()

    def eval_dpo_data(
        self,
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (Union[`Dataset`, Dict[str, `Dataset`]), *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. If it is a dictionary, it will
                evaluate on each dataset, prepending the dictionary key to the metric name. Datasets must implement the
                `__len__` method.

                <Tip>

                If you pass a dictionary with names of datasets as keys and datasets as values, evaluate will run
                separate evaluations on each dataset. This can be useful to monitor how training affects other
                datasets or simply to get a more fine-grained evaluation.
                When used with `load_best_model_at_end`, make sure `metric_for_best_model` references exactly one
                of the datasets. If you, for example, pass in `{"data1": data1, "data2": data2}` for two datasets
                `data1` and `data2`, you could specify `metric_for_best_model="eval_data1_loss"` for using the
                loss on `data1` and `metric_for_best_model="eval_data1_loss"` for the loss on `data2`.

                </Tip>

            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # handle multipe eval datasets
        if dist.get_rank() == 0:
            logger.info('at rank 0, begin eval dpo data')
        def formatted_dict(d: Dict) -> Dict:
            """Format a dictionary for printing."""
            return {k: (f"{v:.5g}" if type(v) == float else v) for k, v in d.items()}

        if True:
            logger.info('here is rank {}, start eval'.format(dist.get_rank()))
            self.model.eval()
            eval_dataset = self.eval_dpo_dataset
            assert eval_dataset != None

            # memory metrics - must set up as early as possible

            eval_data_loader = DataLoader(eval_dataset, batch_size=self.args.per_device_train_batch_size, shuffle=False,collate_fn=self.data_collator)
            eval_data_loader = self.accelerator.prepare(eval_data_loader)


            dpo_forwarder = DPO_Forwarder(self.model, self.reference_model, dist.get_rank(), dist.get_world_size())
            # dpo_forwarder = DPO_Forwarder(self.model, self.reference_model, 0, 1)

            all_eval_metrics = defaultdict(list)
            example_counter = 0
            for j, eval_batch in enumerate(tqdm.tqdm(eval_data_loader,disable=(dist.get_rank()!=0))):
                # rank0_print('start eval_batch {}'.format(j))
                tmp_eval_batch_size = eval_batch['chosen_labels'].size()[0]
                for k, v in eval_batch.items():
                    eval_batch[k] = v.to(self.model.device)
                # rank0_print('eval_batch {} move to cuda'.format(j))
                with torch.no_grad():
                    loss, eval_metrics = dpo_forwarder.get_batch_metrics(eval_batch, self.dpo_args, train=False)

                for k, v in eval_metrics.items():
                    all_eval_metrics[k].extend(v)

                mean_eval_metrics = {k: sum(v) / len(v) for k, v in all_eval_metrics.items()}

                example_counter+=tmp_eval_batch_size
                rank0_print(f'eval after {example_counter}: {formatted_dict(mean_eval_metrics)}')
            self.model.train()
        else:
            logger.info('here is rank {}, waiting for rank0 to finish eval.'.format(dist.get_rank()))
        dist.barrier()

