from dataclasses import dataclass, field
from typing import Optional
import transformers

@dataclass
class LoraArguments:
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    lora_target_modules: str = 'all-linear'
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False

@dataclass
class ModelArguments:
    representation_id: int
    model_name_or_path: Optional[str] = field(default="mistralai/Mistral-7B-v0.1")
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to allow for custom models defined on the Hub in their own modeling files"
        },
    )
    padding_side: str = field(default="right", metadata={"help": "The padding side in tokenizer"})
    representation_token_num: int = field(default=1)
    wrap_q_p: str = field(default=None, metadata={"help": "The value can be `instruction` or `query_or_passage`"})

@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False
    data_root_dir: str = None

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    query_max_length: int = field(default=None)
    doc_max_length: int = field(default=None)
    n_hard_negative: int = field(default=None)
    chunk_sizes: int = field(default=None)
    do_grad_cache: bool = field(default=None)
    temperature: int = field(default=None)
    flash_attention: bool = field(default=None)
    batch_same_task: bool = field(default=None)
