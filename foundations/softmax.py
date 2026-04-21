import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        max_z = max(z)
        sum_z = 0
        for i in range(len(z)):
            sum_z+=math.e**(z[i]-max_z)

        for i in range(len(z)):
            z[i] = round(math.e**(z[i]-max_z) / sum_z , 4)
        return z