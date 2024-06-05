import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


from evaluation_metrics import EvaluationMetrics  


def process_row(metrics, row, embedding, index):
    reference_texts = row['brief_summary']
    generated_text = row['Generated_summary']
    rouge_score = metrics.rouge.get_scores(reference_texts, generated_text)
    
    results = {
        'cosine_similarity': metrics.cosine_similarity(embedding[index][0].detach(), embedding[index][1].detach()),
        'informativeness': metrics.informativeness_score(generated_text, reference_texts),
        'unigram_repetition_rate': metrics.calculate_repetition_rate(generated_text, 1),
        'bigram_repetition_rate': metrics.calculate_repetition_rate(generated_text, 2),
        'trigram_repetition_rate': metrics.calculate_repetition_rate(generated_text, 3),
        'unigram_redundancy': metrics.calculate_redundancy(generated_text, 1),
        'bigram_redundancy': metrics.calculate_redundancy(generated_text, 2),
        'trigram_redundancy': metrics.calculate_redundancy(generated_text, 3),
        'conciseness_score': metrics.calculate_conciseness(generated_text, reference_texts)
    }
    # Add ROUGE scores
    for metric in ['rouge-1', 'rouge-2', 'rouge-l']:
        for score in ['f', 'p', 'r']:
            results[f"{metric}_{score}"] = rouge_score[0][metric][score]
    
    return results




def prepare_dataset(data_a, data_b):
    data_a['combined'] = data_a.apply(lambda row: ', '.join([f"{col}: {row[col]}" for col in data_a.columns if col != 'combined']), axis=1)
    data_a['labels'] = data_b['eligibility/criteria/textblock']
    return data_a[['combined', 'labels']]

def process_row(test_ds, index, generated_criteria_pipeline, tokenizer, model, embedding_list):
    combined_text = test_ds.loc[index, 'combined']

    out = generated_criteria_pipeline(combined_text)
    generated_text = out[0]["summary_text"]
    test_ds.loc[index, 'Generating_Criteria'] = generated_text

    inputs_embedding = get_text_embedding(combined_text, tokenizer, model)
    outputs_embedding = get_text_embedding(generated_text, tokenizer, model)
    embedding_list.append((inputs_embedding, outputs_embedding))

def get_text_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.encoder(inputs["input_ids"])
    return torch.mean(outputs.last_hidden_state, dim=1)

def save_embeddings(embedding_list):
    torch.save(embedding_list, './data/embedding_all.pt')
    embedding_list.clear()
    embedding_list.extend(torch.load('./data/embedding_all.pt'))

def main():
    base_path = "./data"
    test_x = pd.read_csv('./data/test_x.csv').drop('detailed_description/textblock', axis='columns')
    test_y = pd.read_csv('./data/test_y.csv')
    test_ds = prepare_dataset(test_x, test_y)
    output_csv_path = os.path.join(base_path, "Generating_Criteria.csv")
    embedding_path = os.path.join(base_path, "embedding_all.pt")
    model_checkpoint = "./models/flan-t5-base/"
    generated_criteria_pipeline = pipeline('summarization', model=model_checkpoint, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    results_metric_path = os.path.join(base_path, "results_metric.csv") 
    test_ds['Generating_Criteria'] = pd.NA
    embedding_list = []
    metrics = EvaluationMetrics()
    for i in range(len(test_ds)):
        process_row(test_ds, i, generated_criteria_pipeline, tokenizer, model, embedding_list)

        if (i + 1) % 50 == 0:
            save_embeddings(embedding_list)

    save_embeddings(embedding_list)
    test_ds.to_csv(output_csv_path, index=False)


    # Evaluate metrics
    test_x = pd.read_csv(output_csv_path)
    embedding = torch.load(embedding_path)
    results_df = pd.DataFrame(columns=['rouge-1_f', 'rouge-1_p', 'rouge-1_r', 
                                       'rouge-2_f', 'rouge-2_p', 'rouge-2_r', 
                                       'rouge-l_f', 'rouge-l_p', 'rouge-l_r',
                                       'cosine_similarity',
                                       'informativeness',
                                       'unigram_repetition_rate',
                                       'bigram_repetition_rate',
                                       'trigram_repetition_rate',
                                       'unigram_redundancy',
                                       'bigram_redundancy',
                                       'trigram_redundancy',
                                       'conciseness_score'])

    results = [process_row(metrics, row, embedding, index) for index, row in test_x.iterrows()]
    results_df = results_df.append(results, ignore_index=True)

    results_df.to_csv(results_metric_path, index=False)

if __name__ == "__main__":
    main()
