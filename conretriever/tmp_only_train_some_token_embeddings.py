from transformers import AutoModelForCausalLM
import torch
if __name__ == '__main__':
    mistral_model = AutoModelForCausalLM.from_pretrained('/remote-home/share/models/mistral_7b_base')

    for k, param in mistral_model.named_parameters():
        param.requires_grad = False
    mistral_model.model.embed_tokens.requires_grad_()

    mistral_model = mistral_model.to(torch.device('cuda:0'))

    input_ids = torch.tensor([list(range(50))]).to(torch.device('cuda:0'))


    # 方便起见，这里定义id小于10的为special token
    is_special_tokens = (input_ids < 10).to(torch.long).unsqueeze(-1)

    embedding_1 = mistral_model.model.embed_tokens(input_ids).detach()
    embedding_2 = mistral_model.model.embed_tokens(input_ids)

    print('is_special_tokens:{}'.format(is_special_tokens.size()))
    print('embedding_1:{}'.format(embedding_1.size()))
    input_embedding = is_special_tokens * embedding_2 + (1-is_special_tokens) * embedding_1

    # 只训embedding层的所有参数
    # output = mistral_model(input_ids, labels=input_ids)

    # 只训special token对应的embedding参数
    output = mistral_model(inputs_embeds=input_embedding, labels=input_ids)

    output.loss.backward()
    print('mistral_model.model.embed_tokens.rquires_grad:{}'.format(mistral_model.model.embed_tokens.weight.requires_grad))

    # 观察梯度是否正确
    print(mistral_model.model.embed_tokens.weight.grad[:30,:5])


