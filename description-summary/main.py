from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import torch
import os
from evaluation_metrics import EvaluationMetrics  

def text_embedding(input_text, tokenizer, model):
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    outputs = model.encoder(input_ids)
    last_hidden_states = outputs.last_hidden_state
    embedding = torch.mean(last_hidden_states, dim=1)
    return embedding


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

def main():
    # Set relative paths
    base_path = "./data"
    model_path = os.path.join(base_path, "text_summarization")#"Falconsai/text_summarization"
    input_csv_path = os.path.join(base_path, "test.csv")
    output_csv_path = os.path.join(base_path, "generated_summary.csv")
    embedding_path = os.path.join(base_path, "embedding_all.pt")
    results_metric_path = os.path.join(base_path, "results_metric.csv")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    # Load data
    df = pd.read_csv(input_csv_path)
    detailed_description = df['detailed_description']
    print(f"Total descriptions: {len(detailed_description)}")

    # Initialize summarizer
    summarizer = pipeline("summarization", model=model_path, device=7)

    # Initialize columns and lists
    df['Generated_summary'] = pd.NA
    embedding_list = []
    metrics = EvaluationMetrics()

    # Process each description
    for i, description in enumerate(detailed_description):
        summary = summarizer(description)[0]["summary_text"]
        input_embedding = text_embedding(description, tokenizer, model)
        output_embedding = text_embedding(summary, tokenizer, model)

        df.at[i, 'Generated_summary'] = summary
        embedding_list.append((input_embedding, output_embedding))
        
        print(f"Processed {i+1}/{len(detailed_description)}")

        # Save progress every 1000 iterations
        if (i + 1) % 1000 == 0:
            torch.save(embedding_list, embedding_path)
            embedding_list.clear()
            embedding_list = torch.load(embedding_path)

    # Save final embeddings and updated dataframe
    torch.save(embedding_list, embedding_path)
    df.to_csv(output_csv_path, index=False)

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
