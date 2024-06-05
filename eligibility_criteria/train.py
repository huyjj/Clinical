import os
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, 
                          DataCollatorForSeq2Seq, Seq2SeqTrainer, 
                          Seq2SeqTrainingArguments)
import evaluate
import nltk
from nltk.tokenize import sent_tokenize
from huggingface_hub import HfFolder



def load_data(data_type):
    x = pd.read_csv(f'eligibility-criteria-design/{data_type}_x.csv').drop('detailed_description/textblock', axis=1)
    y = pd.read_csv(f'eligibility-criteria-design/{data_type}_y.csv')
    return x, y

def create_dataset(dataa, datab):
    dataa['combined'] = dataa.apply(lambda row: ', '.join([f"{col}: {row[col]}" for col in dataa.columns if col != 'combined']), axis=1)
    dataa['labels'] = datab['eligibility/criteria/textblock']
    return Dataset.from_pandas(dataa)

def preprocess_function(sample, tokenizer, padding="max_length"):
    inputs = ["Generating Criteria: " + item for item in sample["combined"]]
    model_inputs = tokenizer(inputs, max_length=512, padding=padding, truncation=True)
    labels = tokenizer(text_target=sample["labels"], max_length=512, padding=padding, truncation=True)
    if padding == "max_length":
        labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]
    return preds, labels

def compute_metrics(eval_preds, tokenizer, metric):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    result["gen_len"] = np.mean([np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds])
    return result
def main():
    nltk.download('punkt')
    os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3,4'

    train_x, train_y = load_data('train')
    test_x, test_y = load_data('test')

    train_ds = create_dataset(train_x, train_y)
    test_ds = create_dataset(test_x, test_y)

    columns_to_remove = [col for col in train_ds.column_names if col not in ['combined', 'labels']]
    model_id = "/data2/wangyue/LLM/clinical_trail/flan-t5-base"#flan-t5-base
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    tokenized_train_ds = train_ds.map(lambda x: preprocess_function(x, tokenizer), batched=True, remove_columns=columns_to_remove)
    tokenized_test_ds = test_ds.map(lambda x: preprocess_function(x, tokenizer), batched=True, remove_columns=columns_to_remove)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    
    metric = evaluate.load("rouge")
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"{model_id.split('/')[-1]}-trail",
        per_device_train_batch_size=6,
        per_device_eval_batch_size=6,
        predict_with_generate=True,
        fp16=False,
        learning_rate=5e-5,
        num_train_epochs=5,
        logging_dir=f"{model_id.split('/')[-1]}-trail/logs",
        logging_strategy="steps",
        logging_steps=500,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="tensorboard",
        push_to_hub=False,
        hub_strategy="every_save",
        hub_model_id=f"{model_id.split('/')[-1]}-trail",
        hub_token=HfFolder.get_token(),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_train_ds,
        eval_dataset=tokenized_test_ds,
        tokenizer=tokenizer,
        compute_metrics=lambda x: compute_metrics(x, tokenizer, metric),
    )

    trainer.train()

if __name__ == "__main__":
    main()
