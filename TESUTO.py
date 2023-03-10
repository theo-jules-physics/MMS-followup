masses = np.array([1., 1., 1.])
max_f = 1
system = [Ressort(88.7, [0.05, 0.100, 0.150], [0, 1.5], [-1, 1]),
          Ressort(88.7, [0.040, 0.080, 0.100],[0, 1.5], [-1, 1]),
          Ressort(88.7, [0.030, 0.060, 0.105],[0, 1.5], [-1, 1])]
c = 2
dt = 0.1

test_env = Couplage_n_Env_Spring(masses, max_f, system, c, dt)
