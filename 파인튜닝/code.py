import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

conversations = [
    ("안녕하세요!", "안녕하세요!"),
    ("오늘 날씨가 어때요?", "맑고 좋아요."),
    ("뭐 해?", "책을 읽고 있어요."),
    ("뭐 먹을까?", "피자 어때요?"),
    # 추가 대화 데이터
]

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenized_conversations = []

for user_input, bot_response in conversations:
    conversation = f"User: {user_input}\nBot: {bot_response}"
    tokens = tokenizer.encode(conversation, add_special_tokens=True)
    tokenized_conversations.append(tokens)

max_length = max(len(tokens) for tokens in tokenized_conversations)
padded_conversations = [tokens + [tokenizer.pad_token_id] * (max_length - len(tokens)) for tokens in tokenized_conversations]

# None 값을 가진 대화 데이터는 걸러냄
filtered_padded_conversations = [conv for conv in padded_conversations if None not in conv]

if len(filtered_padded_conversations) == 0:
    print("모든 대화 데이터가 None 값을 가지고 있습니다. 데이터를 확인하세요.")
else:
    input_ids = torch.tensor(filtered_padded_conversations)

    # 모델 생성
    model_config = GPT2Config.from_pretrained('gpt2')
    model = GPT2LMHeadModel(config=model_config)

    # 학습
    batch_size = 4
    dataset = CustomDataset(input_ids)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    num_epochs = 3
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            input_batch = batch.to(model.device)
            output = model(input_ids=input_batch, labels=input_batch)
            loss = criterion(output.logits.view(-1, model.config.vocab_size), input_batch.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss:.4f}")

    # 모델 저장
    model_save_path = "fine_tuned_gpt2"
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
