import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def test1():
    ts = pd.Series(np.random.randn(1000),
                   index=pd.date_range('1/1/2000', periods=1000))
    ts = ts.cumsum()
    ts.plot()


if __name__ == "__main__":
    plt.style.use('ggplot')
    plt.close('all')
    test1()
    plt.show()
