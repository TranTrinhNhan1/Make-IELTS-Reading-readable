import numpy as np
import evaluate

# Load cả BLEU và ROUGE từ thư viện evaluate
metric_bleu = evaluate.load("sacrebleu")
metric_rouge = evaluate.load("rouge")

def compute_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds
    
    if isinstance(preds, tuple):
        preds = preds[0]
    
    # Decode kết quả dự đoán
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    # Xử lý labels: thay -100 bằng pad_token_id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Clean up text nhẹ nhàng
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    # Tính BLEU 
    # SacreBLEU yêu cầu references là list of list
    result_bleu = metric_bleu.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])
    
    # Tính ROUGE 
    result_rouge = metric_rouge.compute(predictions=decoded_preds, references=decoded_labels)

    return {
        "bleu": result_bleu["score"],
        "rouge1": result_rouge["rouge1"], # Trùng lặp 1 từ
        "rouge2": result_rouge["rouge2"], # Trùng lặp cụm 2 từ
        "rougeL": result_rouge["rougeL"], # Chuỗi con dài nhất
        "gen_len": np.mean([len(t) for t in decoded_preds])
    }