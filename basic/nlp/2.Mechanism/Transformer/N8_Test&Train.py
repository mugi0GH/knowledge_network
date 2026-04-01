# 超参
d_model = 512
heads = 8
N = 6
src_vocab = len(EN_TEXT.vocab)
trg_vocab = len(FR_TEXT.vocab)

model = Transformer(src_vocab,d_model,N,heads)
for p in model.parameters():
    if p.dim()>1:
        nn.init,xavier_uoniform_(p)

optim = torch.optim.Adam(model.parameters(),lr=0.0001,betas=(0.9,0.98),eps=1e-9)


# 模型训练
def train_model(epochs,print_every=100):
    model.train()
    
    start = time.time()
    temp =start

    total_loss = 0

    for epoch in range(epochs):
        for i, batch in enumerate(train_iter):
            src = batch.English.transpose(0,1)
            trg = batch.Frence.transpose(0,1)
            # 将输入的英语句子中的所有单词翻译成法语
            # 除了最后一个单词，每一位它是结束符

            trg_input = trg[:,:-1]

            # 尝试预测单词
            targets = trg[:,-1:].contiguous().view(-1)
            # 使用掩码代码创建函数来制作掩码
            src_mask, trg_mask = create_masks(src,trg_input)

            preds = model(src,trg_input,src_mask, trg_mask)

            optim.zero_grad()

            loss = F.cross_entropy(preds.view(-1,preds.size(-1)),
                                   results,ignore_index=target_pad)
            loss.backward()
            optim.step()

            total_loss+=loss.data[0]
            if (i+1)% print_every == 0:
                loss_avg = total_loss / print_every
                print("time = %dm, epoch %d, loss = %.3f,%ds per %d iters" % ((time.time()-start)//60,epoch + 1,i +1 ,loss_avg,time.time()-temp,print_every))
                total_loss = 0
                temp = time.time()

# 模型测试
def translate(model, src,max_len = 80, custom_string = False):
    model.eval()
    if custom_sentence == True:
        src = tokenize_en(src)
        sentence = Variable(torch.LongTensor([[EN_TEXT.vocab.stoi[tok] for tok in sentence]])).cuda()

    src_mask = (src != input_pad).unsqueeze(-2)
    e_outputs = model.encoder(src, src_mask)

    outputs = torch.zeros(max_len).type_as(src.data)
    outputs[0] = torch.LongTensor([FR_TEXT.vocab.stoi['<sos>']])

    for i in range(1, max_len):
        trg_mask = np.triu(np.ones((1,i,i),k=1).astype('unit8w'))
        trg_mask = Variable(torch.from_numpy(trg_mask)==0).cuda()

        out = model.out(model.decoder(outputs[:i].unsqueeze(0),e_outputs,src_mask,trg_mask))

        out = F.softmax(out,dim=-1)
        val,ix = out[:,-1].data.topk(1)

        outputs[i] = ix[0][0]

        if ix[0][0] == FR_TEXT.vocab.stoi['<eos>']:
            break
    return ' '.join([FR_TEXT.vocab.itos[ix] for ix in outputs[:i]])