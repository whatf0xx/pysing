from itertools import product
import matplotlib.pyplot as plt
from model import Model


if __name__ == "__main__":
    m = Model(200,
              field=-6.0e-6,
              coupling=5.4e-1)
    s = 4
    t = 5
    fig, axs = plt.subplots(nrows=s, ncols=s, figsize=(9,9))
    for i, j in product(range(s), range(s)):
        m.plot_to_axes(axs[i][j])
        axs[i][j].set_title(f"Step {(s*i+j) * t}")
        # print(f"Beta={m.inverse_temp}")
        # print(m.probability_gradient)
        # print(m.evolution_probs)
        for _ in range(t):
            m.evolve()

    # for _ in range(100):
    #     m.evolve()
    # m.plot_to_axes(axs[s-1][s-1])
    # axs[s-1][s-1].set_title(f"Step {s**2 + 100}")

    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    fig.tight_layout()
    plt.show()
