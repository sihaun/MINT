import torch
import torch.nn.functional as F
from transformers import BertTokenizer, GPT2Tokenizer
from torch.utils.data import DataLoader
from datasets.cifar10 import CIFAR10WithText
from model.Mint import Mint
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 토크나이저 로드
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt_tokenizer.pad_token = gpt_tokenizer.eos_token  # pad_token 세팅

# 모델 초기화
model = Mint().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 데이터셋 및 로더
dataset = CIFAR10WithText(train=True)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

os.makedirs("checkpoints", exist_ok=True)  # 저장 폴더 만들기

best_loss = float("inf")
best_acc = 0.0

model.train()
for epoch in range(5):
    total_correct = 0
    total_tokens = 0
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
    for images, texts in pbar:
        images = images.to(device)

        # 1) 이미지 → 텍스트
        gpt_enc = gpt_tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        input_ids = gpt_enc.input_ids.to(device)
        attention_mask = gpt_enc.attention_mask.to(device)

        out = model.forward_image_to_text(images)
        logits = out.logits
        min_len = min(logits.size(1), input_ids.size(1))

        # 손실 계산
        loss1 = F.cross_entropy(
            logits[:, :min_len, :].reshape(-1, logits.size(-1)),
            input_ids[:, :min_len].reshape(-1),
            ignore_index=gpt_tokenizer.pad_token_id
        )

        # 정확도 계산
        preds = torch.argmax(logits[:, :min_len, :], dim=-1)
        mask = (input_ids[:, :min_len] != gpt_tokenizer.pad_token_id)

        correct = (preds == input_ids[:, :min_len]) & mask
        total_correct += correct.sum().item()
        total_tokens += mask.sum().item()

        # 2) 텍스트 → 이미지
        bert_enc = bert_tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
        '''
        txt_feat = model.text_encoder(input_ids=bert_enc.input_ids, attention_mask=bert_enc.attention_mask).last_hidden_state[:, 0]

        shared_feat = model.shared(txt_feat, reverse=True)
        recon_img = model.image_decoder(shared_feat)'''

        recon_img = model.forward_text_to_image(bert_enc.input_ids, bert_enc.attention_mask)

        loss2 = F.mse_loss(recon_img, images)

        # 최종 손실
        loss = loss1 + loss2
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # tqdm 출력 업데이트
        acc = total_correct / total_tokens * 100 if total_tokens > 0 else 0
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'loss1': f"{loss1.item():.4f}",
            'loss2': f"{loss2.item():.4f}",
            'acc': f"{acc:.2f}%"
        })

    # 체크포인트 저장
    avg_loss = total_loss / len(loader)
    final_acc = total_correct / total_tokens * 100

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "checkpoints/best_loss.pt")

    if final_acc > best_acc:
        best_acc = final_acc
        torch.save(model.state_dict(), "checkpoints/best_acc.pt")

    torch.save(model.state_dict(), "checkpoints/last.pt")
    print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f} | Text Accuracy: {final_acc:.2f}%")
