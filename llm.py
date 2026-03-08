import numpy as np
import sys

np.random.seed(42)

TEXT = """What is an LLM? LLM stands for Large Language Model. In simple terms it is a piece of software that has been trained on massive amounts of text from the internet, books, articles, code, and conversations. It learned patterns in how humans use language by reading more text than any person could read in a thousand lifetimes. It does not think. It predicts. Given what you just said, it calculates the most likely next word, then the next, then the next. It does this so well that it feels like a conversation but underneath it is math."""

CONTEXT_LEN = 8
EMBED_DIM   = 32
LR          = 0.005
EPOCHS      = 4000
BATCH       = 32
D           = EMBED_DIM

chars  = sorted(set(TEXT))
VOCAB  = len(chars)
ch2id  = {c:i for i,c in enumerate(chars)}
id2ch  = {i:c for i,c in enumerate(chars)}
encode = lambda s: [ch2id[c] for c in s]
decode = lambda ids: "".join(id2ch[i] for i in ids)

data = encode(TEXT)
X, Y = [], []
for i in range(len(data) - CONTEXT_LEN):
    X.append(data[i : i+CONTEXT_LEN])
    Y.append(data[i+1 : i+CONTEXT_LEN+1])
X, Y = np.array(X), np.array(Y)

s = lambda *sh: np.random.randn(*sh) * 0.05
W = {
    'E':    s(VOCAB, D),
    'P':    s(CONTEXT_LEN, D),
    'Wq':   s(D, D),
    'Wk':   s(D, D),
    'Wv':   s(D, D),
    'Wo':   s(D, D),
    'W1':   s(D, 2*D),
    'b1':   np.zeros(2*D),
    'W2':   s(2*D, D),
    'b2':   np.zeros(D),
    'Wout': s(D, VOCAB),
    'bout': np.zeros(VOCAB),
}

def forward(idx):
    B, T  = idx.shape
    cache = {}
    x     = W['E'][idx] + W['P'][np.arange(T)]
    cache['x0'] = x.copy()

    Q = x @ W['Wq']; K = x @ W['Wk']; V = x @ W['Wv']
    scores = Q @ K.transpose(0,2,1) / D**0.5
    scores = np.clip(scores, -10, 10)
    scores += np.triu(np.ones((T,T)), k=1) * -1e9
    e = np.exp(scores - scores.max(-1, keepdims=True))
    A = e / e.sum(-1, keepdims=True)
    attn_out = (A @ V) @ W['Wo']
    x2 = x + attn_out
    cache.update({'A':A, 'V':V, 'Q':Q, 'K':K, 'attn_out':attn_out, 'x2':x2})

    h  = np.maximum(0, x2 @ W['W1'] + W['b1'])
    ff = h @ W['W2'] + W['b2']
    x3 = x2 + ff
    cache.update({'h':h, 'x3':x3, 'idx':idx})
    return x3 @ W['Wout'] + W['bout'], cache

def loss_and_grad(logits, targets):
    B, T, V = logits.shape
    shifted = logits - logits.max(-1, keepdims=True)
    probs   = np.exp(shifted) / np.exp(shifted).sum(-1, keepdims=True)
    loss    = -np.log(probs[np.arange(B)[:,None], np.arange(T)[None,:], targets] + 1e-9).mean()
    dlogits = probs.copy()
    dlogits[np.arange(B)[:,None], np.arange(T)[None,:], targets] -= 1
    dlogits /= (B * T)
    return loss, dlogits

def backward(dlogits, cache):
    B, T  = cache['idx'].shape
    grads = {k: np.zeros_like(v) for k, v in W.items()}

    grads['Wout'] = cache['x3'].reshape(-1,D).T @ dlogits.reshape(-1,VOCAB)
    grads['bout'] = dlogits.sum(axis=(0,1))
    dx3  = dlogits @ W['Wout'].T
    dff  = dx3
    grads['W2'] = cache['h'].reshape(-1,2*D).T @ dff.reshape(-1,D)
    grads['b2'] = dff.sum(axis=(0,1))
    dh   = dff @ W['W2'].T
    dhr  = dh * (cache['h'] > 0)
    grads['W1'] = cache['x2'].reshape(-1,D).T @ dhr.reshape(-1,2*D)
    grads['b1'] = dhr.sum(axis=(0,1))
    dx2  = dx3 + dhr @ W['W1'].T
    grads['Wo'] = cache['attn_out'].reshape(-1,D).T @ dx2.reshape(-1,D)
    da   = dx2 @ W['Wo'].T
    dV   = cache['A'].transpose(0,2,1) @ da
    dA   = da @ cache['V'].transpose(0,2,1)
    ds   = cache['A'] * (dA - (dA * cache['A']).sum(-1, keepdims=True)) / D**0.5
    dQ   = ds @ cache['K']
    dK   = ds.transpose(0,2,1) @ cache['Q']
    dx0  = da @ W['Wo'].T + dV @ W['Wv'].T + dQ @ W['Wq'].T + dK @ W['Wk'].T
    np.add.at(grads['E'], cache['idx'], dx0)
    grads['P']  = dx0.sum(axis=0)
    grads['Wv'] = cache['x0'].reshape(-1,D).T @ dV.reshape(-1,D)
    grads['Wq'] = cache['x0'].reshape(-1,D).T @ dQ.reshape(-1,D)
    grads['Wk'] = cache['x0'].reshape(-1,D).T @ dK.reshape(-1,D)

    # gradient clipping — stops numbers exploding at high epochs
    for key in grads:
        np.clip(grads[key], -1.0, 1.0, out=grads[key])

    return grads

def generate(seed, n=150):
    ids = encode(seed)
    for _ in range(n):
        ctx = ids[-CONTEXT_LEN:]
        while len(ctx) < CONTEXT_LEN:
            ctx = [ch2id[' ']] + ctx
        logits, _ = forward(np.array([ctx]))
        last  = logits[0, -1]
        e     = np.exp(last - last.max())
        probs = e / e.sum()
        ids.append(np.random.choice(VOCAB, p=probs))
    return decode(ids)

print("Training...\n")
for epoch in range(EPOCHS):
    idx = np.random.permutation(len(X))
    for i in range(0, len(X), BATCH):
        xb, yb        = X[idx[i:i+BATCH]], Y[idx[i:i+BATCH]]
        logits, cache = forward(xb)
        loss, dlogits = loss_and_grad(logits, yb)
        grads         = backward(dlogits, cache)
        for key in W:
            W[key] -= LR * grads[key]
    if epoch % 200 == 0 or epoch == EPOCHS - 1:
        print(f"  Epoch {epoch:>4}  |  loss = {loss:.4f}")

seed = sys.argv[1] if len(sys.argv) > 1 else "LLM"
print(f"\n{generate(seed, 150)}")
