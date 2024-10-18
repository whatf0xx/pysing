from itertools import product
import matplotlib.pyplot as plt
from model import Model


if __name__ == "__main__":
    fig, axs = plt.subplots(3, 3)
    for i, j in product(range(3), repeat=2):
        h = float(f"1.0e-{3-i}")
        k = float(f"{j+3}.8e-1")
        m = Model(100,
                  field=h,
                  coupling=k)
        for _ in range(50):
            m.evolve()

        m.plot_to_axes(axs[i][j])
        axs[i][j].set_title(f"H={h}; J={k}")

    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    fig.tight_layout()
    plt.show()
