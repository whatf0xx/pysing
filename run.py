from itertools import product
import matplotlib.pyplot as plt
from tqdm.contrib.itertools import product
from model import Model


if __name__ == "__main__":
    fig, axs = plt.subplots(3, 3)
    for i, j, k in product(range(3), range(3), range(80)):
        if k == 0:
            h = float(f"{1+4*i}.0e-2")
            k = float(f"{j+3}.4e-1")
            m = Model(250, field=h, coupling=k)
        m.evolve()

        m.plot_to_axes(axs[i][j])
        axs[i][j].set_title(
            f"$H={h:.2f}; T={m.critical_temp:.2f} T_c$", fontsize=8
        )

    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    fig.suptitle("Ising model microstates for varying $(H, T)$", fontsize=10)
    fig.tight_layout()
    plt.savefig(fname="exampel.png", dpi=320)
    plt.show()
