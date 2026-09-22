import numpy as np

x = np.array([1, 2, 3])
print("x shape:", x.shape)
res = np.tile(x, (1, 2))
print("res shape:", res.shape)
print("res:", res)
res2 = res[:, :5][0]
print("res2:", res2)

res3 = np.tile(x, (2,))[:5]
print("res3:", res3)
