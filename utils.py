import os
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


def scan_feature(dir_path, save_path):
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.feature'):
                with open(os.path.join(root, file), 'r',encoding='utf8') as f:
                    content = f.read()
                with open(os.path.join(save_path, file,), 'w',encoding='utf8') as f:
                    f.write(content)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def cal_sim(sentence1, sentence2):
    sentences = [sentence1, sentence2]
    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    # Compute cosine-similarity
    cosine_similarity = F.cosine_similarity(sentence_embeddings[0], sentence_embeddings[1], dim=0)
    return cosine_similarity



if __name__ == '__main__':
    # for file in os.listdir('dataset'):
    #     # 如果 os.path.join('user_story_feature', file) 不存在，创建
    #     if not os.path.exists(os.path.join('user_story_feature', file)):
    #         os.mkdir(os.path.join('user_story_feature', file))
    #
    #     scan_feature(os.path.join('./dataset', file), os.path.join('user_story_feature', file))
    print(cal_sim('I sort students', 'I sort teachers'))

