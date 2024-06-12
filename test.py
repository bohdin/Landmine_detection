import numpy as np
i = 0
np.random.seed(42)
while i < 10:
    print(np.random.randint(1, 100), end=" ")
    i += 1