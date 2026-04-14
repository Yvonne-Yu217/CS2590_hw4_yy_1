import os
import sys
import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records
import torch.nn as nn

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load best model
print("Loading best model...")
model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-small')
checkpoint_path = 'checkpoints/ft_experiments/experiment/best_model.pt'
model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')

# Load data
print("Loading data...")
_, dev_loader, test_loader = load_t5_data(16, 16)

# Dev evaluation
print("Evaluating on dev set...")
total_loss = 0
total_tokens = 0
criterion = nn.CrossEntropyLoss(ignore_index=-100)
all_dev_predictions = []

with torch.no_grad():
    for encoder_input, encoder_mask, decoder_input, decoder_targets, initial_decoder_inputs in tqdm(dev_loader):
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        logits = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input,
        )['logits']

        non_pad = decoder_targets != -100
        loss = criterion(logits.view(-1, logits.size(-1)), decoder_targets.view(-1))
        num_tokens = torch.sum(non_pad).item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

        generated = model.generate(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            max_length=512,
            num_beams=4,
            early_stopping=True,
        )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        all_dev_predictions.extend(decoded)

eval_loss = total_loss / total_tokens
print(f"Dev loss: {eval_loss}")

# Save dev predictions and compute metrics
gt_sql_path = 'data/dev.sql'
gt_record_path = 'records/ground_truth_dev.pkl'
model_sql_path = 'results/t5_ft_experiment_dev.sql'
model_record_path = 'records/t5_ft_experiment_dev.pkl'

save_queries_and_records(all_dev_predictions, model_sql_path, model_record_path)
sql_em, record_em, record_f1, error_msgs = compute_metrics(gt_sql_path, model_sql_path, gt_record_path, model_record_path)
error_rate = sum(1 for msg in error_msgs if msg) / len(error_msgs) if error_msgs else 0

print(f"Dev set results: Loss: {eval_loss}, Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}")
print(f"Dev set results: {error_rate*100:.2f}% of the generated outputs led to SQL errors")

# Test inference
print("\nRunning test inference...")
all_test_predictions = []
with torch.no_grad():
    for encoder_input, encoder_mask, initial_decoder_inputs in tqdm(test_loader):
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)

        generated = model.generate(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            max_length=512,
            num_beams=4,
            early_stopping=True,
        )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        all_test_predictions.extend(decoded)

model_sql_path_test = 'results/t5_ft_experiment_test.sql'
model_record_path_test = 'records/t5_ft_experiment_test.pkl'
save_queries_and_records(all_test_predictions, model_sql_path_test, model_record_path_test)
print(f"Test outputs saved to {model_sql_path_test} and {model_record_path_test}")
print("Done!")
