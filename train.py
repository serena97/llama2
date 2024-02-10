from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import data

model_dir = "./llama/llama-2-7b-chat-hf"
peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1, target_modules=["q_proj", "v_proj"])
model = LlamaForCausalLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

train_dataset = data.TrainDataset()
collator = DataCollatorForSeq2Seq(train_dataset.tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)


training_args = TrainingArguments(
    output_dir="trained/llama-2-7b-chat-hf/mt0-large-lora",
    learning_rate=1e-3,
    per_device_train_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
    evaluation_strategy="no",
    save_strategy="epoch",
    save_steps=200,
    eval_steps=None,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=collator,
)

trainer.train()