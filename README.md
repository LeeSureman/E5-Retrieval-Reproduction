# ConRetriever


This repository contatins the whole pipeline for reproducing the LLM-based dense retriever [E5-Mistral-7b-instruct](https://huggingface.co/intfloat/e5-mistral-7b-instruct), including training data, training and evaluation code.

## Contents

- [Install](#install)
- [Training and Evaluation](#train_eval)
- [Checkpoint](#checkpoint)
- [Synthetic-Training-Data](#Synthetic-Training-Data)
- [Acknowledgement](#acknowledgement)

## Install

1. Clone this repository and navigate to the ConRetriever folder
```bash
git clone https://github.com/LeeSureman/E5-Retrieval-Reproduction
cd ConRetriever
```

2. Install Package
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Train_Eval

For simplicity, we place the training code and testing code in the same script named [train_eval_llm_retriever.sh](./scripts/train_eval_llm_retriever.sh).

In the script, we use `mistralai/Mistral-7B-v0.1` as an example for illustration. Before beginning training, two very important steps must be completed: data preparation and determination of the transformer layer.

### Data Preparation

For training data, we use the following format:
```python
# Assuming there are data for two tasks in a folder named `demo`
demo
|--- task1.jsonl
|--- task2.jsonl

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

Then, set the sampling ratio, query type, and message type in the file of [task_config.py](./conretriever/task_config.py).

### Determine the transformer layer

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

## Checkpoint

We have released a model checkpoint fine-tuned from [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1), using data converted by the script [hf_to_training_data.py](./hf_to_training_data.py). You can access to the checkpoint [here](https://huggingface.co/BeastyZ/e5-R-mistral-7b)🤗.

## Synthetic-Training-Data

If you want to use synthetic data to train a model as a retriever, you need to first generate synthetic data using 1) [brainstorm_task.sh](./scripts/brainstorm_task.sh) and 2) [generate_examples.sh](./scripts/generate_examples.sh). We follow [[Wang et al., 2023]](https://arxiv.org/abs/2401.00368) to generate synthetic data.

## Acknowledgement

For training, we use [GradCache](https://github.com/luyug/GradCache) to enable contrastive learning training with large batch size.

For evaluation, we use [e5](https://github.com/microsoft/unilm/tree/9c0f1ff7ca53431fe47d2637dfe253643d94185b/e5) to evaluate the performance of the model.
