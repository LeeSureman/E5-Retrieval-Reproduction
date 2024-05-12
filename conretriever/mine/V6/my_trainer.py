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

import torch.nn as nn

if is_peft_available():
    from peft import PeftModel

from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from modeling_retriever import retrieval_forward
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
        # print(features)
        batch_size = len(features)
        # print('features:{}'.format(len(features)))
        # print('features keys:{}'.format(features[0].keys()))

        batch_outputs = {}
        for i in range(batch_size):
            inputs = features[i]
            for key, value in inputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        # for k,v in batch_outputs.items():
        #     batch_outputs[k] = torch.cat(torch.tensor(v),dim=0)

        result = BatchEncoding(batch_outputs, tensor_type=self.return_tensors)
        # for k,v in result.items():
        #     print('{}: {}'.format(k,v.size()))
        return result
        # batch = self.tokenizer.pad(
        #     features,
        #     padding=self.padding,
        #     max_length=self.max_length,
        #     pad_to_multiple_of=self.pad_to_multiple_of,
        #     return_tensors=self.return_tensors,
        # )
        # if "label" in batch:
        #     batch["labels"] = batch["label"]
        #     del batch["label"]
        # if "label_ids" in batch:
        #     batch["labels"] = batch["label_ids"]
        #     del batch["label_ids"]
        # return batch


class My_Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        # outputs = model(**inputs)
        if 'hard_negative_docs_ids' not in inputs:
            inputs['hard_negative_docs_ids'] = None
            inputs['hard_negative_docs_attention_mask'] = None

        outputs = retrieval_forward(model,
                                    inputs['query_ids'], inputs['query_attention_mask'],
                                    inputs['positive_doc_ids'], inputs['positive_doc_attention_mask'],
                                    inputs['hard_negative_docs_ids'], inputs['hard_negative_docs_attention_mask'],
                                    chunk_sizes=self.chunk_sizes, do_grad_cache=self.args.do_grad_cache,
                                    n_hard_negative=self.args.n_hard_negative,temperature=self.args.temperature)

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
        self.data_collator = MyDataCollatorWithPadding(self.tokenizer)

    # def training_step(self, *args, **kwargs):
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        if self.args.batch_same_task:
            # check batch is from the same task
            if dist.get_rank() in [0,-1]:
                tid = inputs['task_id'][0]
                tid_same = torch.mean((inputs['task_id'] == tid).to(torch.float))
                assert tid_same.item() == 1, 'task id in the local batch is not the same'
                task = self.train_dataset.tid_to_task[tid]
                print('task_id: {}, task: {}'.format(tid, task))
        if self.args.batch_same_task:
            # check batch_same_task
            # print("inputs['task_id']: {}".format(inputs['task_id']))
            # print('query_ids: {}'.format(inputs['query_ids'].size()))
            # whole_wolrd_task_ids = gather_tensor(inputs['task_id'])
            # print('whole_wolrd_task_ids:{}'.format(whole_wolrd_task_ids.size()))
            # whole_wolrd_task_ids_is_same = (whole_wolrd_task_ids == whole_wolrd_task_ids[0]).to(torch.float)
            # whole_wolrd_task_ids_is_same = torch.mean(whole_wolrd_task_ids_is_same)
            # print('whole_wolrd_task_ids_is_same:{}'.format(whole_wolrd_task_ids_is_same.size()))
            # assert whole_wolrd_task_ids_is_same.item() == 1, 'task id in the global batch is not the same'
            pass



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
        if self.args.batch_same_task:
            return SequentialSampler(self.train_dataset)
        else:
            return super()._get_train_sampler()

    # def get_train_dataloader(self, *args, **kwargs):
    #     if self.args.batch_same_task:
    #         return self.get_train_dataloader_for_batch_same_task()
    #     else:
    #         return super().get_train_dataloader(*args, **kwargs)
    #

    # def create_accelerator_and_postprocess(self):
    #     grad_acc_kwargs = {"num_steps": self.args.gradient_accumulation_steps}
    #     grad_acc_kwargs["sync_with_dataloader"] = False
    #     gradient_accumulation_plugin = GradientAccumulationPlugin(**grad_acc_kwargs)
    #
    #     # create accelerator object
    #     accelerator_kwargs = {}
    #     if self.args.accelerator_config is not None:
    #         accelerator_kwargs = self.args.accelerator_config
    #         # dict and AcceleratorConfigs are parseable, json files are not
    #         if isinstance(accelerator_kwargs, AcceleratorConfig):
    #             accelerator_kwargs = accelerator_kwargs.to_dict()
    #         elif isinstance(accelerator_kwargs, dict):
    #             # Some values may need to go through non-accelerate aligned defaults
    #             # and we need to run the `__post_init__` to set them
    #             accelerator_kwargs = AcceleratorConfig(**accelerator_kwargs).to_dict()
    #     print('accelerator_kwargs:')
    #     for k,v in accelerator_kwargs.items():
    #         print('{}: {}'.format(k, v))
    #
    #     self.accelerator = Accelerator(
    #         deepspeed_plugin=self.args.deepspeed_plugin,
    #         gradient_accumulation_plugin=gradient_accumulation_plugin,
    #         **accelerator_kwargs,
    #     )
    #     # some Trainer classes need to use `gather` instead of `gather_for_metrics`, thus we store a flag
    #     self.gather_function = self.accelerator.gather_for_metrics
    #
    #     # deepspeed and accelerate flags covering both trainer args and accelerate launcher
    #     self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
    #     self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
    #
    #     # post accelerator creation setup
    #     if self.is_fsdp_enabled:
    #         fsdp_plugin = self.accelerator.state.fsdp_plugin
    #         fsdp_plugin.limit_all_gathers = self.args.fsdp_config.get(
    #             "limit_all_gathers", fsdp_plugin.limit_all_gathers
    #         )
    #         if is_accelerate_available("0.23.0"):
    #             fsdp_plugin.activation_checkpointing = self.args.fsdp_config.get(
    #                 "activation_checkpointing", fsdp_plugin.activation_checkpointing
    #             )
    #             if fsdp_plugin.activation_checkpointing and self.args.gradient_checkpointing:
    #                 raise ValueError(
    #                     "The activation_checkpointing in FSDP config and the gradient_checkpointing in training arg "
    #                     "can't be set to True simultaneously. Please use FSDP's activation_checkpointing logic "
    #                     "when using FSDP."
    #                 )
    #
    #     if self.is_deepspeed_enabled and getattr(self.args, "hf_deepspeed_config", None) is None:
    #         self.propagate_args_to_deepspeed()
    #
    #     # `save_only_model` can't be used with DeepSpeed/FSDP along with `load_best_model_at_end`
    #     if (
    #         self.args.save_only_model
    #         and (self.is_deepspeed_enabled or self.is_fsdp_enabled)
    #         and self.args.load_best_model_at_end
    #     ):
    #         wrapper = "DeepSpeed" if self.is_deepspeed_enabled else "FSDP"
    #         raise ValueError(f"{wrapper} can't be used with `save_only_model` along with `load_best_model_at_end`.")
    #
    #     # `auto_find_batch_size` isn't yet supported with DeepSpeed/FSDP
    #     if (self.is_deepspeed_enabled or self.is_fsdp_enabled) and self.args.auto_find_batch_size:
    #         wrapper = "DeepSpeed" if self.is_deepspeed_enabled else "FSDP"
    #         raise NotImplementedError(f"`{wrapper}` doesn't support `auto_find_batch_size`.")
