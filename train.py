import torch.optim as optim

from translators import Config
from translators.model_builder import build_model, build_tokenizer, build_dataset, build_criterion
from tqdm import tqdm

if __name__ == "__main__":
    cnf = Config("examples/GNMT_Config.yaml", "GNMT")

    tokenizer = build_tokenizer(cnf)
    model = build_model(cnf)
    dataset = build_dataset(cnf, tokenizer)
    criterion = build_criterion(cnf.vocab_size, tokenizer.vocab.stoi[tokenizer.pad_token], cnf.device)
    optimizer = optim.Adam(model.parameters(), lr=cnf.learning_rate)
    train_iter = dataset['train'].iter_dataset()
    for epoch in range(cnf.epochs):
        avg_loss = 0
        for batch in tqdm(train_iter):
            src, src_length = batch.src
            tgt, tgt_length = batch.tgt
            output = model(src, src_length, tgt[:, :-1])
            tgt_labels = tgt[:, 1:]
            T, B = output.size(1), output.size(0)
            loss = criterion(output.view(T * B, -1), tgt_labels.contiguous().view(-1))
            loss_per_batch = loss.item()
            loss /= B
            avg_loss += loss
            loss.backward()
            optimizer.step()
            model.zero_grad()
        print(f'Loss: {avg_loss}')