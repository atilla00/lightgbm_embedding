# LightGBM Embeddings

Feature embeddings with LightGBM


## Examples
```python
df = pd.read_csv(
    "https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv"
)
cols = df.columns[:-1]
target = df.columns[-1]
num_classes = df[target].nunique()

X_train, X_test = train_test_split(
    df, test_size=0.2, stratify=df[target], random_state=42
)

n_dim = 20
emb = LightgbmEmbedding(n_dim=n_dim)
emb.fit(X_train[cols], X_train[target])
X_train_embed = emb.transform(X_train[cols])
X_test_embed = emb.transform(X_test[cols])
```
