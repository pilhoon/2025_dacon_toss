
import numpy as np, pandas as pd
from sklearn.datasets import make_classification

def gen(n=200000, f=10, seed=42):
    X, y = make_classification(n_samples=n, n_features=f, n_informative=int(0.6*f),
                               n_redundant=int(0.2*f), weights=[0.97,0.03], random_state=seed)
    t = np.arange(n)
    df = pd.DataFrame(X, columns=[f'f{i}' for i in range(f)])
    df['label'] = y.astype(int)
    df['time'] = t
    return df

if __name__ == '__main__':
    df = gen()
    df.to_parquet('synthetic.parquet')
    print('Wrote synthetic.parquet', df.shape)
