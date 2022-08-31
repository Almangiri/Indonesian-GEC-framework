

# Trained models
We provide four versions for L2R and R2L of our proposed model that we used for re-ranking L2R as below: 

- [IGEC_R2L.pth](https://drive.google.com/file/d/15J2qPWy2-x8rSejSyxoE6fYrRebkaBsD/view?usp=sharing)


- [IGEC_L2R.pth](https://drive.google.com/file/d/15J2qPWy2-x8rSejSyxoE6fYrRebkaBsD/view?usp=sharing)



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
