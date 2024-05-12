from vllm import LLM, SamplingParams
import torch

def vllm_inference_func(model_name_or_path, prompts, temperature, top_p, max_new_tokens, max_model_len):
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens)
    llm = LLM(model=model_name_or_path, tensor_parallel_size=torch.cuda.device_count(),
              max_model_len=1024)

    outputs_vllm = llm.generate(prompts, sampling_params)

    outputs_texts = []

    for output in outputs_vllm:
        outputs_texts.append(output.outputs[0].text)
    
    return outputs_texts
    pass
if __name__ == '__main__':


    prompts = [
        "Tell me a 400-word story.",
        "Who is taylor swift?"
    ]

    prompts = prompts * 100

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95,max_tokens=512)

    llm = LLM(model="/remote-home/share/models/mistral_7b_instruct", tensor_parallel_size=4, max_model_len=1024)

    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")