# E5-Retrieval-Reproduction

This repository contatins the whole pipeline for reproducing the LLM-based dense retriever [E5-Mistral-7b-instruct](https://huggingface.co/intfloat/e5-mistral-7b-instruct), including training data, training and evaluation code.

## Contents

- [Environment](#environment)
- [Training-Evaluation](#training-evaluation)
- [Checkpoint](#checkpoint)
- [Synthetic-Training-Data](#synthetic-training-data)
- [Acknowledgement](#acknowledgement)

## Environment

1. Clone this repository and navigate to the E5-Retrieval-Reproduction folder
```bash
git clone https://github.com/LeeSureman/E5-Retrieval-Reproduction.git
cd E5-Retrieval-Reproduction
```

2. Install Package
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Training-Evaluation

### Quick Reproduction

If you do not consider the details and just want to quickly reproduce our results, simply execute the following commands in sequence.

```bash
# 1. This will download the training data to the folder `training_data/reproduction`
python download_training_data.py

# 2. After downloading the training data, you can run the following command to train and evaluate the LLM-based dense retriever. The checkpoint will be saved to the folder `conretriever/checkpoint_dir/reproduction`.
bash scripts/reproduction.sh
```

### Detailed Configuration

Here, we show you how to reproduce our results or train other LLM-based dense retriever through detailed configuration. For simplicity, we will demonstrate you through a `demo`. The script for running `demo` is [train_eval_llm_retriever.sh](./scripts/train_eval_llm_retriever.sh). In the script, we use `mistralai/Mistral-7B-v0.1` as an example for illustration. Before training, three very important steps must be completed: **data preparation**, **determination of the transformer layer**, and **task configuration**.

#### Data Preparation

For training data, we use the following format:
```python
# Now there are two tasks in the folder `training_data/demo`
demo
|--- demo.jsonl
|--- synthetic.jsonl

# For each task, the data follows the following format (If the task is not "synthetic", the instruction field can be omitted)
"""
{
    "query": "What are some popular Italian pasta recipes?",
    "positive": [
        "Italian cuisine is known for its diverse range of pasta dishes. From classic favorites like spaghetti carbonara and fettuccine alfredo to regional specialties like lasagna and ravioli, Italian pasta recipes offer a wide variety of flavors and ingredients. One popular recipe is penne arrabbiata, which is made with penne pasta, a spicy tomato sauce, garlic, and red chili flakes. Another delicious option is tortellini with pesto sauce, where homemade tortellini pasta is filled with a mixture of cheese and served with a flavorful basil pesto sauce. For seafood lovers, linguine with clams is a must-try dish, featuring linguine pasta tossed with fresh clams, garlic, white wine, and parsley. Additionally, pasta primavera is a delightful vegetarian option made with mixed vegetables, cream, and Parmesan cheese. These are just a few examples of the countless Italian pasta recipes that you can explore and enjoy."
    ],
    "negative": [
        "Italian cuisine is famous for its delectable pasta dishes. One of the most popular pasta recipes is spaghetti carbonara, which originated in Rome and features pasta tossed with a creamy egg and pancetta sauce. Another classic Italian dish is fettuccine alfredo, where fettuccine noodles are coated in a rich Parmesan cheese sauce. Lasagna is another beloved Italian pasta dish, made with layers of pasta, meat sauce, and cheese. Additionally, ravioli is a traditional Italian pasta dish consisting of stuffed pasta pockets served with various sauces. Italian pasta recipes are loved worldwide for their simplicity, fresh ingredients, and bold flavors."
    ],
    "instruction": "Given a food cuisine, retrieve recipes or restaurant reviews from that cuisine. "
}
"""
```
Please check the sample data for more information: [demo.jsonl](./training_data/demo/demo.jsonl) and [synthetic.jsonl](./training_data/demo/synthetic.jsonl).

#### Determine the transformer layer

Correctly set the value of parameter `fsdp_transformer_layer_cls_to_wrap` in the [train_eval_llm_retriever.sh](./scripts/train_eval_llm_retriever.sh) file. Use the following code to get the correct value. Here, the transformer layer is the `MistralDecoderLayer` in the Mistral model. **If you use other models, you can also use the same method to determine the transformer layer.**
```python
from transformers import AutoModelForCausalLM

model_name_or_path = 'mistralai/Mistral-7B-v0.1'
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
print(model)

"""Output:
MistralForCausalLM(
  (model): MistralModel(
    (embed_tokens): Embedding(32000, 4096)
    (layers): ModuleList(
      (0-31): 32 x MistralDecoderLayer(
        (self_attn): MistralSdpaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): MistralRotaryEmbedding()
        )
        (mlp): MistralMLP(
          (gate_proj): Linear(in_features=4096, out_features=14336, bias=False)
          (up_proj): Linear(in_features=4096, out_features=14336, bias=False)
          (down_proj): Linear(in_features=14336, out_features=4096, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): MistralRMSNorm()
        (post_attention_layernorm): MistralRMSNorm()
      )
    )
    (norm): MistralRMSNorm()
  )
  (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
)
"""
```

#### Task configuration

Set the sampling weight, query type, and message type in the file of [task_config.py](./conretriever/task_config.py).


After finishing the above three steps, you can run the command `bash scripts/train_eval_llm_retriever.sh` to train and evaluate the `demo`. The checkpoint will be saved to the folder `conretriever/checkpoint_dir/demo`.

## Checkpoint

We have released a model checkpoint fine-tuned from [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1), using data converted by the script [download_training_data.py](./download_training_data.py). You can get the checkpoint [here](https://huggingface.co/BeastyZ/e5-R-mistral-7b)🤗. Below is an example to encode queries and passages from the MS-MARCO passage ranking dataset.
```python
mport torch
from torch import Tensor
from typing import List, Mapping
from transformers import AutoTokenizer, AutoModel
from transformers.modeling_outputs import BaseModelOutput


def specific_token_pool(
    outputs: BaseModelOutput,
    batch_dict: Tensor,
    representation_id: int = 2,
    representation_token_num: int = 1
) -> Tensor:
    with torch.cuda.amp.autocast():
        input_ids = batch_dict['input_ids']
        attention_mask = batch_dict['attention_mask']
        tmp_batch_size = input_ids.size()[0]
        seq_len = input_ids.size()[1]

        is_representation_id = (input_ids == representation_id)
        range_tensor = torch.arange(seq_len).unsqueeze(0).to(is_representation_id.device)
        seq_len = torch.sum(attention_mask, dim=1)
        first_representation_token_pos = seq_len - (representation_token_num)
        mask = range_tensor < (first_representation_token_pos.unsqueeze(1))
        # mask the representation_token in the original input
        is_representation_id[mask] = False

        last_hidden_states = outputs.last_hidden_state
        hidden_size = last_hidden_states.size()[-1]

        sequence_representation_embeds = last_hidden_states[is_representation_id]
        sequence_representation_embeds = sequence_representation_embeds.view(tmp_batch_size, -1, hidden_size)
        sequence_representation = torch.mean(sequence_representation_embeds, dim=1)
        sequence_representation = torch.nn.functional.normalize(sequence_representation, p=2, dim=-1)
        return sequence_representation
    

def create_batch_dict(
    input_texts: List[str],
    tokenizer: AutoTokenizer, 
    representation_id: int = 2,
    representation_token_num: int = 1, 
    max_length: int = 512,
    return_token_type_ids: bool = False,
    truncation: bool = True,
):
    representation_special_ids = [representation_id for _ in range(representation_token_num)]

    batch_dict = tokenizer(
        input_texts,
        max_length=max_length - representation_token_num,
        return_token_type_ids=return_token_type_ids,
        return_attention_mask=False,
        padding=False,
        truncation=truncation
    )

    # append eos_token_id to every input_ids
    batch_dict['input_ids'] = [input_ids + representation_special_ids for input_ids in batch_dict['input_ids']]
    return tokenizer.pad(
        batch_dict,
        padding=True,
        pad_to_multiple_of=8,
        return_attention_mask=True,
        return_tensors="pt",
    )


def move_to_cuda(sample):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda(non_blocking=True)
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_cuda(value) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return tuple([_move_to_cuda(x) for x in maybe_tensor])
        elif isinstance(maybe_tensor, Mapping):
            return type(maybe_tensor)({k: _move_to_cuda(v) for k, v in maybe_tensor.items()})
        else:
            return maybe_tensor

    return _move_to_cuda(sample)


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'{task_description}\n{query}'


max_length = 512
representation_token_num = 1
representation_id = 2

# Each query must come with a one-sentence instruction that describes the task
task = 'Given a web search query, retrieve relevant passages that answer the query'
queries = [
    get_detailed_instruct(task, 'how much protein should a female eat'),
    get_detailed_instruct(task, 'summit define')
]
# No need to add instruction for retrieval documents
documents = [
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments."
]

tokenizer = AutoTokenizer.from_pretrained('/remote-home/ctzhu/github_repo/ConRetriever-v2/ConRetriever/public_weight_mistral', trust_remote_code=True)
model = AutoModel.from_pretrained('/remote-home/ctzhu/github_repo/ConRetriever-v2/ConRetriever/public_weight_mistral', trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()

tokenizer.pad_token = tokenizer.unk_token

query_batch_dict = create_batch_dict(queries, 
                                     tokenizer,
                                     max_length=max_length,
                                     representation_id=representation_id,
                                     representation_token_num=representation_token_num)
doc_batch_dict = create_batch_dict(documents, 
                                   tokenizer,
                                   max_length=max_length,
                                   representation_id=representation_id, 
                                   representation_token_num=representation_token_num)

query_batch_dict = move_to_cuda(query_batch_dict)
doc_batch_dict = move_to_cuda(doc_batch_dict)

query_batch_dict['output_hidden_states']=False
doc_batch_dict['output_hidden_states']=False
query_outputs = model(**query_batch_dict)
doc_outputs = model(**doc_batch_dict)

query_embeddings = specific_token_pool(query_outputs, query_batch_dict)
doc_embeddings = specific_token_pool(doc_outputs, doc_batch_dict)

scores = (query_embeddings @ doc_embeddings.T) * 100
print(scores.tolist())

```

## Synthetic-Training-Data

If you want to manually synthesize the training data for retriever, you can run these scripts 1) [brainstorm_task.sh](./scripts/brainstorm_task.sh) and 2) [generate_examples.sh](./scripts/generate_examples.sh). We follow [[Wang et al., 2023]](https://arxiv.org/abs/2401.00368) to generate synthetic data.

## Acknowledgement

For training, we use [GradCache](https://github.com/luyug/GradCache) to enable contrastive learning training with large batch size.

For evaluation, we use [e5](https://github.com/microsoft/unilm/tree/9c0f1ff7ca53431fe47d2637dfe253643d94185b/e5) to evaluate the performance of the model.
