#V2:加上切换成lora训练的功能
#V3:加上对hard negative训练的支持
#V4: 加上对多检索数据集的支持，每个检索数据集的query端有对应的prompt
#V5: 把representation token改成和RepLLAMA一模一样的 （去掉representation token前面的eos和空格）
#V6：支持在多数据集检索时所有数据来自同一个任务，支持带伪文章训练，以及伪文章生成，打分，DPO训练等。
#V7：相比V6删去多余部分，只保留检索器训练的部分。

