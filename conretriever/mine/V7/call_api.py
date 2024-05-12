import requests


def call_mistral(prompt, max_new_tokens=512, num_return_sequences=8) -> str:
    """Call self-constructed api or vllm-based api."""
    # vllm-based api
    url = "http://slurmd-7:30803/generate"
    data = {
        "prompt": prompt,
        "max_tokens": max_new_tokens,
        "n": num_return_sequences,
        "temperature": 0.0
    }
    response = requests.post(url, json=data)
    print(response)
    return response.json()['text'][0].split("[/INST]")[-1].strip()


if __name__ == "__main__":
    prompt_template = """<s>[INST] Please write a passage to answer the question
# Question: {prompt}
# Passage:[/INST]"""
    text = "Hello, I'm Beasty from China. Who are you?"
    prompt = prompt_template.format(prompt=text)
    res = call_mistral(
        prompt,
        max_new_tokens=512,
        num_return_sequences=1,
    )
    print(res)

"""
说明：
模型部署完成后，后台具体如何接收API请求，请看 https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/api_server.py 中的 generate 函数

请求服务的具体参数请看 https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py 中的 SamplingParams 类

当前的 url 为 "http://slurmd-7:30803/generate"，表示模型部署在 slurmd-7 节点上，调用API服务时必须确保当前终端没有开启VPN服务，或者设置 slurmd-7 不走代理（export no_proxy="slurmd-7"）
"""