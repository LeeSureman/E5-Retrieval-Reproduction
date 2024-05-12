from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import jsonlines
import tqdm
import torch.distributed as dist

class My_Dataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, i):
        result = self.tokenizer(self.data[i],
                                max_length=128,
                                padding='max_length',
                                return_token_type_ids=False,
                                return_attention_mask=False,
                                truncation=True,
                                return_tensors="pt"
                                )
        result['input_ids'] = result['input_ids'][0]
        return result

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    accelerator = Accelerator()
    device = accelerator.device

    mistral_tokenizer = AutoTokenizer.from_pretrained('/remote-home/share/models/mistral_7b_instruct',
                                                      )

    mistral_tokenizer.pad_token = mistral_tokenizer.unk_token
    tmp = list(jsonlines.open('retrieval_data/multi_dataset_2024_2_13/nq.jsonl'))
    raw_data = []
    for tmp_js in tmp:
        pos = tmp_js['positive'][0]
        raw_data.append(pos)


    my_dataset = My_Dataset(raw_data, mistral_tokenizer)
    if dist.get_rank() == 0:
        print(my_dataset[0])

    mistral_model = AutoModelForCausalLM.from_pretrained('/remote-home/share/models/mistral_7b_instruct',
                                                         torch_dtype=torch.float16)
    mistral_model = mistral_model.to(device)


    my_data_loader= torch.utils.data.DataLoader(my_dataset, shuffle=True, batch_size=16)
    mistral_model, my_data_loader = accelerator.prepare(mistral_model, my_data_loader)
    mistral_model.eval()
    with torch.no_grad():
        for batch in tqdm.tqdm(my_data_loader):
            # if dist.get_rank() == 0:
            #     print(batch)
            batch = batch.to(device)
            mistral_model(**batch)



