import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import os.path as op
import json
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, RobertaModel, RobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaLMHead
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, SequenceClassifierOutput


# ------------------------------------------------------------------------------
# Utility components for EasyRec (simplified for inference only)
# ------------------------------------------------------------------------------
class MLPLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features):
        return self.activation(self.dense(features))


class Pooler(nn.Module):
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"]

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return (last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
        elif self.pooler_type == "avg_first_last":
            first_hidden = outputs.hidden_states[1]
            last_hidden = outputs.hidden_states[-1]
            pooled = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled
        elif self.pooler_type == "avg_top2":
            second_last = outputs.hidden_states[-2]
            last = outputs.hidden_states[-1]
            pooled = ((last + second_last) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled
        else:
            raise NotImplementedError


class EasyRecEncoder(RobertaPreTrainedModel):
    """Minimal EasyRec encoder for inference only (no contrastive training)."""
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, pooler_type='cls'):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.pooler_type = pooler_type
        self.pooler = Pooler(pooler_type)
        if pooler_type == "cls":
            self.mlp = MLPLayer(config)
        self.init_weights()

    def encode(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        pooled = self.pooler(attention_mask, outputs)
        if self.pooler_type == "cls":
            pooled = self.mlp(pooled)
        return BaseModelOutputWithPoolingAndCrossAttentions(
            pooler_output=pooled,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
        )


# ------------------------------------------------------------------------------
# Text encoder wrappers (all inherit from nn.Module and implement forward)
# ------------------------------------------------------------------------------
class BaseEncoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, texts):
        raise NotImplementedError


class BGE(BaseEncoder):
    def __init__(self, device):
        super().__init__(device)
        self.tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')
        self.model = AutoModel.from_pretrained('BAAI/bge-large-en-v1.5').to(device)
        self.model.eval()

    def forward(self, texts):
        inputs = self.tokenizer(texts.tolist(), padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :]
            return F.normalize(cls_emb, p=2, dim=1)


class Blair(BaseEncoder):
    def __init__(self, device):
        super().__init__(device)
        self.tokenizer = AutoTokenizer.from_pretrained("hyp1231/blair-roberta-base")
        self.model = AutoModel.from_pretrained("hyp1231/blair-roberta-base").to(device)
        self.model.eval()

    def forward(self, texts):
        inputs = self.tokenizer(texts.tolist(), padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :]
            return F.normalize(cls_emb, p=2, dim=1)


class BERT(BaseEncoder):
    def __init__(self, device):
        super().__init__(device)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased').to(device)
        self.model.eval()

    def forward(self, texts):
        inputs = self.tokenizer(texts.tolist(), padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :]
            return F.normalize(cls_emb, p=2, dim=1)


class RoBERTa_large_sentence(BaseEncoder):
    def __init__(self, device):
        super().__init__(device)
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.model = RobertaModel.from_pretrained('roberta-large').to(device)
        self.model.eval()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, texts):
        inputs = self.tokenizer(texts.tolist(), padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            sent_emb = self.mean_pooling(outputs, inputs['attention_mask'])
            return F.normalize(sent_emb, p=2, dim=1)


class GTE_7B(BaseEncoder):
    def __init__(self, device):
        super().__init__(device)
        self.tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-Qwen2-7B-instruct', trust_remote_code=True)
        self.model = AutoModel.from_pretrained('Alibaba-NLP/gte-Qwen2-7B-instruct', trust_remote_code=True).to(device)
        self.model.eval()

    def last_token_pool(self, last_hidden_states, attention_mask):
        sequence_lengths = attention_mask.sum(dim=1) - 1
        return last_hidden_states[torch.arange(last_hidden_states.size(0), device=last_hidden_states.device), sequence_lengths]

    def forward(self, texts):
        inputs = self.tokenizer(texts.tolist(), padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            pooled = self.last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
            return F.normalize(pooled, p=2, dim=1)


class EasyRec(BaseEncoder):
    def __init__(self, device):
        super().__init__(device)
        self.config = AutoConfig.from_pretrained("hkuds/easyrec-roberta-large")
        self.model = EasyRecEncoder.from_pretrained("hkuds/easyrec-roberta-large", config=self.config).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("hkuds/easyrec-roberta-large", use_fast=False)
        self.model.eval()

    def forward(self, texts):
        inputs = self.tokenizer(texts.tolist(), padding=True, truncation=True, max_length=512, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model.encode(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        return F.normalize(outputs.pooler_output, p=2, dim=1)

class LLM2VecEncoder(BaseEncoder):
    """LLM2Vec encoder for Qwen2-0.5B (or any LLM2Vec model)."""
    def __init__(self, device, model_name_or_path="McGill-NLP/LLM2Vec-Qwen2-0.5B-mntp-unsup-simcse"):
        super().__init__(device)
        try:
            from llm2vec import LLM2Vec
        except ImportError:
            raise ImportError(
                "llm2vec is required for LLM2VecEncoder. "
                "Install it with `pip install llm2vec` and `pip install flash-attn --no-build-isolation`"
            )
        # Load the LLM2Vec model. You may also pass peft_model_name_or_path if needed.
        self.model = LLM2Vec.from_pretrained(
            model_name_or_path,
            device_map=device if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()

    def forward(self, texts):
        # texts is a list of strings
        # LLM2Vec.encode expects a list of sentences, returns numpy array or tensor
        with torch.no_grad():
            embeddings = self.model.encode(texts.tolist())
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings).to(self.device)
        return embeddings

ENCODER_REGISTRY = {
    'BGE': BGE,
    'Blair': Blair,
    'BERT': BERT,
    'RoBERTa_large_sentence': RoBERTa_large_sentence,
    'GTE_7B': GTE_7B,
    'EasyRec': EasyRec,
    'LLM2Vec': LLM2VecEncoder,
}


# ------------------------------------------------------------------------------
# Embedding extraction
# ------------------------------------------------------------------------------
DATASET_PATH_MAP = {
    "Games_5core": "Video_Games/5-core/downstream",
    "Movies_5core": "Movies_and_TV/5-core/downstream",
    "Arts_5core": "Arts_Crafts_and_Sewing/5-core/downstream",
    "Sports_5core": "Sports_and_Outdoors/5-core/downstream",
    "Baby_5core": "Baby_Products/5-core/downstream",
    "Goodreads": "Goodreads/clean",
}


def extract_item_embeddings(encoder, dataset_name, batch_size, prompt_type, save_info=None):
    raw_path = DATASET_PATH_MAP[dataset_name]
    with open(f"./data/{raw_path}/item_titles.json", 'r', encoding='utf-8') as f:
        item_metadata = json.load(f)

    item_ids = [int(k) for k in item_metadata.keys()]
    max_item_id = max(item_ids)
    assert 0 not in item_ids

    item_titles = ["Null"] + [item_metadata[str(i)] for i in range(1, max_item_id + 1)]
    item_infos = np.array(item_titles)

    if prompt_type == "title":
        prompts = item_infos
    elif prompt_type == "direct":
        instruct = "To recommend this item to users, this item can be described as: "
        prompts = np.char.add(instruct, item_infos)
    else:
        raise ValueError(f"Unknown prompt_type: {prompt_type}")

    item_embeds = []
    for i in tqdm(range(0, len(prompts), batch_size), desc=f"Extracting {prompt_type}"):
        batch = prompts[i:i + batch_size]
        with torch.no_grad():
            embeds = encoder(batch).cpu()
        item_embeds.append(embeds)
    item_embeds = torch.cat(item_embeds, dim=0).numpy()

    save_dir = f"./item_info/{dataset_name}"
    os.makedirs(save_dir, exist_ok=True)
    model_name = save_info if save_info else type(encoder).__name__
    save_path = op.join(save_dir, f"{model_name}_{prompt_type}_item_embs.npy")
    np.save(save_path, item_embeds)
    print(f"Saved embeddings to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, choices=list(ENCODER_REGISTRY.keys()))
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--prompt_type", type=str, default="title", choices=["title", "direct"])
    parser.add_argument("--save_info", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    encoder_cls = ENCODER_REGISTRY[args.model_name]
    encoder = encoder_cls(device)
    encoder.eval()

    extract_item_embeddings(encoder, args.dataset, args.batch_size, args.prompt_type, args.save_info)


if __name__ == "__main__":
    main()
