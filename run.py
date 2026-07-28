import matplotlib.pyplot as plt
from tqdm import trange
from model import Model


if __name__ == "__main__":
    # Reduced units: J is the energy scale and T is measured in J / k_B, so
    # the critical point of the infinite lattice sits at T = 2.269. These
    # values reproduce the dimensionless parameters of the original demo,
    # beta*J = 0.3 and beta*H = 1.0, i.e. T ~ 1.47 T_c.
    coupling = 1.0
    temperature = 10 / 3
    field = 10 / 3
    boundary = "open"
    field_time = 10
    relax_time = 50

    m = Model(
              300,
              temperature=temperature,
              field=field,
              coupling=coupling,
              boundary=boundary,
              seed=0,
          )
    magnetisation = [m.magnetisation_per_spin]
    for i in trange(field_time + relax_time):
        if i == field_time:
            m.H = 0
        m.evolve()
        magnetisation.append(m.magnetisation_per_spin)

    fig, ax = plt.subplots()
    ax.plot(range(field_time + relax_time + 1), magnetisation, "b.")
    ax.axvline(field_time, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("sweeps")
    ax.set_ylabel("magnetisation per spin")
    ax.set_title(
        f"T = {temperature:.2f} ({m.reduced_temperature:.2f} T_c), "
        f"field removed after {field_time} sweeps"
    )
    plt.show()
