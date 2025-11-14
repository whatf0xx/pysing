import matplotlib.pyplot as plt
from tqdm import trange
from model import Model


if __name__ == "__main__":
    k = 3e-1
    h = 1
    field_time = 10
    relax_time = 50
    m = Model(
              300,
              field=h,
              coupling=k,
          )
    magnetisation = [m.magnetisation]
    for i in trange(field_time + relax_time):
        if i == field_time:
            m.H = 0
        m.evolve()
        magnetisation.append(m.magnetisation)

    fig, ax = plt.subplots()
    ax.plot(range(field_time+relax_time+1), magnetisation, "b.")
    plt.show()

