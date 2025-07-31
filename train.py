from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
import pandas as pd
from datasets import Dataset, DatasetDict
import os

# تحميل البيانات
df = pd.read_csv("training_data.csv")
dataset = Dataset.from_pandas(df)

# تحميل النموذج والتوكنايزر
model_name = "distilbert/distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.getenv("HUGGING_FACE_TOKEN"))
model = AutoModelForCausalLM.from_pretrained(model_name, token=os.getenv("HUGGING_FACE_TOKEN"))
tokenizer.pad_token = tokenizer.eos_token

# معالجة البيانات
def preprocess_function(examples):
    tokenized_inputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)
train_dataset_dict = DatasetDict({"train": tokenized_dataset})

# إعداد التدريب
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1, 
    per_device_train_batch_size=4,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_dict["train"],
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
)

trainer.train()
model.save_pretrained("./elasfar-AI")
tokenizer.save_pretrained("./elasfar-AI")