from itertools import product
import matplotlib.pyplot as plt
from model import Model


if __name__ == "__main__":
    m = Model(200,
              field=1.0e-3,
              coupling=5.4e-1)
    n = Model(200,
              field=1.0e-2,
              coupling=5.4e-1)
    p = Model(200,
              field=1.0e-1,
              coupling=5.4e-1)

    m1 = Model(200,
              field=1.0e-3,
              coupling=3.4e-1)
    n1 = Model(200,
              field=1.0e-2,
              coupling=3.4e-1)
    p1 = Model(200,
              field=1.0e-1,
              coupling=3.4e-1)


    for _ in range(50):
        m.evolve()
        n.evolve()
        p.evolve()
        m1.evolve()
        n1.evolve()
        p1.evolve()

    fig, ax = plt.subplots(2, 3)
    m.plot_to_axes(ax[0][0])
    n.plot_to_axes(ax[0][1])
    p.plot_to_axes(ax[0][2])
    m1.plot_to_axes(ax[1][0])
    n1.plot_to_axes(ax[1][1])
    p1.plot_to_axes(ax[1][2])

    ax[0][0].set_title("H=1.0e-3; J=5.4e-1")
    ax[0][1].set_title("H=1.0e-2; J=5.4e-1")
    ax[0][2].set_title("H=1.0e-1; J=5.4e-1")
    ax[1][0].set_title("H=1.0e-3; J=3.4e-1")
    ax[1][1].set_title("H=1.0e-2; J=3.4e-1")
    ax[1][2].set_title("H=1.0e-1; J=3.4e-1")

    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    fig.tight_layout()
    plt.show()
