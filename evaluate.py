import argparse
import datetime
import os
import json
import random
import hashlib
from pathlib import Path
from collections import defaultdict, OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

# ------------------------------------------------------------------------------
# Default configuration
# ------------------------------------------------------------------------------
DEFAULT_CONFIG = {
    'model': 'SASRec',
    'hidden_size': 64,
    'num_heads': 1,
    'num_blocks': 2,
    'dropout': 0.2,
    'max_seq_length': 50,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'train_batch_size': 256,
    'eval_batch_size': 256,
    'num_epochs': 150,
    'eval_interval': 5,
    'patience': 20,
    'loss_type': 'ce',
    'metrics': ['NDCG', 'HR'],
    'topk': [5, 10],
    'val_metric': 'NDCG@10',
    'rand_seed': 2026,
    'reproducibility': True,
    'device': 'cuda',
    'ckpt_dir': './ckpt',
    'save': True,
}

DATASET_PATH_MAP = {
    "Goodreads": "Goodreads/clean",
    "Games_5core": "Video_Games/5-core/downstream",
    "Movies_5core": "Movies_and_TV/5-core/downstream",
    "Arts_5core": "Arts_Crafts_and_Sewing/5-core/downstream",
    "Sports_5core": "Sports_and_Outdoors/5-core/downstream",
    "Baby_5core": "Baby_Products/5-core/downstream",
}


# ------------------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------------------
def init_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_seed(seed, reproducibility=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_command_line_args(args):
    config = {}
    for arg in args:
        if arg.startswith('--'):
            key, value = arg[2:].split('=', 1)
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
            config[key] = value
    return config

def get_checkpoint_filename(config, suffix='.pth'):
    param_str = json.dumps({k: config[k] for k in ['model', 'dataset', 'lr', 'hidden_size', 'dropout', 'rand_seed'] if k in config}, sort_keys=True)
    hash_str = hashlib.md5(param_str.encode()).hexdigest()[:6]
    return f"{config['model']}_{config['dataset']}_{hash_str}{suffix}"


# ------------------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------------------
class SequenceDataset(Dataset):
    def __init__(self, config, sequences):
        self.sequences = sequences
        self.max_len = config['max_seq_length']

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        item_seq = seq[:-1]
        label = seq[-1]
        seq_len = len(item_seq)
        if seq_len < self.max_len:
            item_seq = item_seq + [0] * (self.max_len - seq_len)
        else:
            item_seq = item_seq[-self.max_len:]
        return {
            'item_seqs': torch.tensor(item_seq, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long),
            'seq_lengths': min(seq_len, self.max_len)
        }

def load_data(config):
    dataset = config['dataset']
    base_path = Path('data') / DATASET_PATH_MAP[dataset]

    def read_seqs(mode=''):
        with open(base_path / f'{mode}data.txt', 'r') as f:
            seqs = [list(map(int, line.split()))[-config['max_seq_length']-1:] for line in f]
        return seqs

    train_seqs = read_seqs('train_')
    valid_seqs = read_seqs('val_')
    test_seqs = read_seqs('test_')
    all_seqs = read_seqs('')

    flat_items = [item for seq in all_seqs for item in seq]
    total_items = max(flat_items) if flat_items else 0

    train_set = SequenceDataset(config, train_seqs)
    valid_set = SequenceDataset(config, valid_seqs)
    test_set = SequenceDataset(config, test_seqs)

    return train_set, valid_set, test_set, total_items


# ------------------------------------------------------------------------------
# SASRec Model
# ------------------------------------------------------------------------------
class SASRec(nn.Module):
    def __init__(self, config, pretrained_item_embeddings=None):
        super().__init__()
        self.config = config
        self.hidden_size = config['hidden_size']
        self.max_seq_len = config['max_seq_length']
        self.item_num = config['item_num']
        self.dropout_rate = config['dropout']
        self.loss_type = config.get('loss_type', 'ce')

        if pretrained_item_embeddings is not None:
            self.item_embedding = nn.Embedding.from_pretrained(pretrained_item_embeddings, freeze=True)
            if pretrained_item_embeddings.shape[1] != self.hidden_size:
                self.emb_proj = nn.Linear(pretrained_item_embeddings.shape[1], self.hidden_size)
            else:
                self.emb_proj = nn.Identity()
        else:
            self.item_embedding = nn.Embedding(self.item_num + 2, self.hidden_size, padding_idx=0)
            self.emb_proj = nn.Identity()

        self.pos_embedding = nn.Embedding(self.max_seq_len, self.hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=config['num_heads'],
            dim_feedforward=self.hidden_size * 4,
            dropout=self.dropout_rate,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config['num_blocks'])
        self.dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm = nn.LayerNorm(self.hidden_size)

    def forward(self, batch):
        item_seqs = batch['item_seqs']
        seq_lengths = batch['seq_lengths']
        labels = batch['labels']

        mask = (item_seqs == 0)
        item_emb = self.item_embedding(item_seqs)
        item_emb = self.emb_proj(item_emb)

        positions = torch.arange(item_seqs.size(1), device=item_seqs.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)
        seq_emb = self.dropout(item_emb + pos_emb)

        seq_emb = self.transformer(seq_emb, src_key_padding_mask=mask)
        seq_emb = self.layer_norm(seq_emb)

        last_indices = (seq_lengths - 1).clamp(min=0)
        last_hidden = seq_emb[torch.arange(seq_emb.size(0)), last_indices]

        if hasattr(self.emb_proj, 'weight'):
            item_weights = self.emb_proj(self.item_embedding.weight)
        else:
            item_weights = self.item_embedding.weight
        logits = torch.matmul(last_hidden, item_weights.transpose(0, 1))

        if self.loss_type == 'ce':
            loss = F.cross_entropy(logits, labels, ignore_index=0)
        elif self.loss_type == 'bpr':
            loss = F.cross_entropy(logits, labels, ignore_index=0)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        return {'loss': loss, 'logits': logits}

    def predict(self, batch, n_return_sequences=10):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(batch)
            logits = outputs['logits']
            logits[:, 0] = -float('inf')
            _, topk_indices = torch.topk(logits, n_return_sequences, dim=-1)
        return topk_indices


MODEL_REGISTRY = {'SASRec': SASRec}


# ------------------------------------------------------------------------------
# Evaluator
# ------------------------------------------------------------------------------
class Evaluator:
    def __init__(self, config):
        self.metrics = config['metrics']
        self.topk = config['topk']
        self.maxk = max(self.topk)

    def calculate_metrics(self, predictions, labels):
        results = {}
        for k in self.topk:
            topk_preds = predictions[:, :k]
            hits = (topk_preds == labels.unsqueeze(1)).any(dim=1).float()
            results[f'HR@{k}'] = hits

        for k in self.topk:
            topk_preds = predictions[:, :k]
            matches = (topk_preds == labels.unsqueeze(1))
            ranks = torch.arange(1, k+1, device=predictions.device).float().unsqueeze(0)
            dcg = (matches.float() / torch.log2(ranks + 1)).sum(dim=1)
            idcg = 1.0 / torch.log2(torch.tensor(2.0, device=predictions.device))
            ndcg = dcg / idcg
            results[f'NDCG@{k}'] = ndcg
        return results


# ------------------------------------------------------------------------------
# Trainer
# ------------------------------------------------------------------------------
class Trainer:
    def __init__(self, config, model, saved_ckpt_path):
        self.config = config
        self.model = model
        self.device = config['device']
        self.evaluator = Evaluator(config)
        self.saved_model_ckpt = saved_ckpt_path
        self.best_metric = -float('inf')
        self.best_epoch = 0

    def train(self, train_loader, valid_loader):
        optimizer = AdamW(self.model.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        n_epochs = self.config['num_epochs']
        patience_counter = 0

        for epoch in range(1, n_epochs + 1):
            self.model.train()
            total_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{n_epochs}")
            for batch in pbar:
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                optimizer.zero_grad()
                outputs = self.model(batch)
                loss = outputs['loss']
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch} training loss: {avg_loss:.6f}")

            if epoch % self.config['eval_interval'] == 0:
                val_results = self.evaluate(valid_loader, split='val')
                val_score = val_results[self.config['val_metric']]
                print(f"Validation results at epoch {epoch}: {val_results}")

                if val_score > self.best_metric:
                    self.best_metric = val_score
                    self.best_epoch = epoch
                    patience_counter = 0
                    torch.save(self.model.state_dict(), self.saved_model_ckpt)
                    print(f"New best model saved (score: {val_score:.6f})")
                else:
                    patience_counter += 1
                    if patience_counter >= self.config['patience']:
                        print(f"Early stopping at epoch {epoch}")
                        break

        print(f"Training finished. Best epoch: {self.best_epoch}, best {self.config['val_metric']}: {self.best_metric:.6f}")

    def evaluate(self, dataloader, split='test'):
        self.model.eval()
        all_results = defaultdict(list)
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {split}"):
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                preds = self.model.predict(batch, n_return_sequences=self.evaluator.maxk)
                metrics = self.evaluator.calculate_metrics(preds, batch['labels'])
                for k, v in metrics.items():
                    all_results[k].append(v)

        output = OrderedDict()
        for metric in self.config['metrics']:
            for k in self.config['topk']:
                key = f"{metric}@{k}"
                output[key] = torch.cat(all_results[key]).mean().item()
        return output


# ------------------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------------------
class Runner:
    def __init__(self, model_name, config_dict=None):
        self.config = DEFAULT_CONFIG.copy()
        if config_dict:
            self.config.update(config_dict)
        self.config['model'] = model_name

        self.config['device'] = init_device()
        init_seed(self.config['rand_seed'], self.config['reproducibility'])

        train_set, valid_set, test_set, total_items = load_data(self.config)
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set
        self.config['item_num'] = total_items

        pretrained_emb = None
        if self.config.get('embedding'):
            emb_path = self.config['embedding']
            pretrained_emb = torch.tensor(np.load(emb_path), dtype=torch.float32).to(self.config['device'])

        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}")
        self.model = MODEL_REGISTRY[model_name](self.config, pretrained_emb)
        self.model.to(self.config['device'])

        os.makedirs(self.config['ckpt_dir'], exist_ok=True)
        ckpt_filename = get_checkpoint_filename(self.config)
        self.saved_ckpt_path = os.path.join(self.config['ckpt_dir'], ckpt_filename)

        self.trainer = Trainer(self.config, self.model, self.saved_ckpt_path)

    def run(self):
        train_loader = DataLoader(self.train_set, batch_size=self.config['train_batch_size'], shuffle=True)
        valid_loader = DataLoader(self.valid_set, batch_size=self.config['eval_batch_size'], shuffle=False)
        test_loader = DataLoader(self.test_set, batch_size=self.config['eval_batch_size'], shuffle=False)

        self.trainer.train(train_loader, valid_loader)
        self.model.load_state_dict(torch.load(self.saved_ckpt_path, map_location=self.config['device']))
        print(f"Loaded best checkpoint from {self.saved_ckpt_path}")

        test_results = self.trainer.evaluate(test_loader, split='test')
        print("Test results:", test_results)

        if not self.config.get('save', True):
            if os.path.exists(self.saved_ckpt_path):
                os.remove(self.saved_ckpt_path)

        return test_results, self.config


# ------------------------------------------------------------------------------
# Main evaluation script
# ------------------------------------------------------------------------------
def calculate_mean_and_std(results_list):
    metrics = {}
    for res in results_list:
        for k, v in res.items():
            metrics.setdefault(k, []).append(v)
    stats = {m: (float(np.mean(vals)), float(np.std(vals))) for m, vals in metrics.items()}
    return stats

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='SASRec')
    parser.add_argument('--dataset', type=str, default='Games_5core')
    parser.add_argument('--embedding', type=str, default='', help='Path to .npy embeddings')
    return parser.parse_known_args()

def main():
    args, unparsed_args = parse_args()
    cmd_config = parse_command_line_args(unparsed_args)
    config = {**vars(args), **cmd_config}

    exp_seeds = [2026]
    test_results = []

    for seed in exp_seeds:
        config['rand_seed'] = seed
        runner = Runner(model_name=args.model, config_dict=config)
        test_res, final_config = runner.run()
        test_results.append(test_res)

    stats = calculate_mean_and_std(test_results)

    timestamp = datetime.datetime.now().strftime('%b-%d-%Y_%H-%M-%S')
    emb_name = os.path.basename(final_config.get('embedding', 'noemb')).replace('.npy', '')
    result_dir = (f"./Results/{final_config['dataset']}/{final_config['model']}/"
                  f"lr_{final_config['lr']}_dr_{final_config['dropout']}_time_{timestamp}_emb_{emb_name}")
    os.makedirs(result_dir, exist_ok=True)

    with open(os.path.join(result_dir, 'results.txt'), 'w') as f:
        f.write(f"Final Results for {final_config['model']} on {final_config['dataset']}:\n")
        for metric, (mean, std) in stats.items():
            f.write(f"{metric}: {mean:.6f} ± {std:.6f}\n")
        f.write("\nPer-seed results:\n")
        for i, res in enumerate(test_results):
            f.write(f"Seed {exp_seeds[i]}:\n")
            for k, v in res.items():
                f.write(f"  {k}: {v:.6f}\n")
            f.write("\n")

    with open(os.path.join(result_dir, 'config.json'), 'w') as f:
        json.dump(final_config, f, indent=4)

    print("Evaluation finished.")
    print(stats)

if __name__ == "__main__":
    main()