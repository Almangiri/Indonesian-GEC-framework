

# Trained models
We provide the best versions of our proposed model in the following link: 

- [IndoGEC.pth](https://drive.google.com/file/d/1NLjHD2ItLx68DQOYQuH1DvvzhnrNM74d/view?usp=sharing)



# load models
To load one of these models in your notebook used the below lines: 

```py
transformer = Transformer(
    encoder=EncoderLayer(
        vocab_size=len(SRC.vocab),
        max_len=MAX_LEN,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        hidden_size=HIDDEN_SIZE,
        dropout=DROPOUT,
        n_layers=N_LAYERS
    ),
    decoder=DecoderLayer(
        vocab_size=len(TRG.vocab),
        max_len=MAX_LEN,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        hidden_size=HIDDEN_SIZE,
        dropout=DROPOUT,
        n_layers=N_LAYERS
    ),
    src_pad_index=SRC.vocab.stoi[SRC.pad_token],
    dest_pad_index=TRG.vocab.stoi[TRG.pad_token]
).to(DEVICE)

optimizer = optim.Adam(params=transformer.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=TRG.vocab.stoi[TRG.pad_token])
trainer = Trainer(model=transformer, optimizer=optimizer, criterion=criterion)

checkpoint = torch.load('./IGEC_R2L.pth')
optimizer.load_state_dict(checkpoint['optimizer'])          
transformer.load_state_dict(checkpoint['state_dict'])
```
The whole code files will be released soon!
