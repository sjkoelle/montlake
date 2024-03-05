from sympy import sin, cos

import numpy as np
from scipy.stats import special_ortho_group

from sym_manifold.sym_ops import Var, Constant
from sym_manifold.sym_ops import custom, concat, elementwise, reshape
from sym_manifold.sym_manifold import CoordSystem

from TSLassoData import TSLassoData


def sample_m1():

    # values used for paper experiment
    A = Constant(np.array([[-2.224596482726371, -0.07313352038651444],
                           [-0.13431766850153554, -1.0062494435296525]]))
    b = Constant(np.array([0.1244304928696931, 1.402807402444574]))

    # A = Constant(np.stack([np.array([np.random.uniform(-2.0, 2.0), np.random.uniform(-0.5, 0.5)]),
    #                        np.array([np.random.uniform(-np.pi/36, np.pi/36),
    #                                  (np.random.uniform(-np.pi/3, -np.pi/5) if np.random.rand() > 0.5 else
    #                                   np.random.uniform(np.pi/5, np.pi/3))])]))
    # b = Constant(np.array([np.random.uniform(-1.0, 1.0), np.random.uniform((5/12) * np.pi, (7/12) * np.pi)]))

    R = Constant(special_ortho_group.rvs(48)[:, :3])
    Rt = Constant(R.numpy_value.T)

    x = Var(name="x", shape=(2, ))
    o = Var(name="o", shape=(48, ))

    h = A @ x + b

    y_small = custom(h, ["1.0/(exp(h_0) + 1.0)", "cos(h_1)", "sin(2.0 * h_1)"])
    y_small_inv = y_small.invert(input_name="ys", selection_values=(-1, 1), num_selection_values=100)

    if isinstance(y_small_inv, list):
        y_small_inv = y_small_inv[0]

    y_small_diff = y_small.to_custom().diff()
    y_small_inv_diff = y_small_inv.to_custom().diff()

    small_coord_system = CoordSystem(d=2, D=3, coord_domain=(-1, 1), coord_func=y_small_inv, inv_func=y_small,
                                     coord_jac=y_small_inv_diff, inv_jac=y_small_diff)

    y = R @ y_small
    y_inv = y.invert(input_name="y", selection_values=(-1, 1), num_selection_values=100)

    if isinstance(y_inv, list):
        y_inv = y_inv[0]

    y_diff = (R @ y_small.to_custom()).diff()
    y_inv_diff = (y_small_inv.to_custom(Rt @ o)).diff()

    full_coord_system = CoordSystem(d=2, D=48, coord_domain=(-1, 1), coord_func=y_inv, inv_func=y,
                                    coord_jac=y_inv_diff, inv_jac=y_diff)

    return small_coord_system, full_coord_system


def sample_m2():

    # values used for paper experiment
    A = Constant(np.array([[-1.8851977431865239, 0.11138430303363234],
                           [0.006830803092539861, 1.2667849053051967]]))
    b = Constant(np.array([0.3702812303981724, -0.45769351026735927]))

    # A = Constant(np.random.random((2, 2)))
    # b = Constant(np.random.random((2,)))

    R = Constant(special_ortho_group.rvs(48)[:, :4])
    Rt = Constant(R.numpy_value.T)

    x = Var(name="x", shape=(2, ))
    o = Var(name="o", shape=(48, ))

    h1 = A @ x + b

    h2 = custom(h1, ["1.0/(1.0 + exp(h_0))", "exp(h_1)"])
    h3 = custom(h2, ["h_0 + 0.5", "2 * h_1"])
    h4 = custom(h3, ["h_0 * cos(h_1)", "h_0 * sin(h_1)"])

    y_small = concat([h2, h4])
    y_small_inv = y_small.invert(input_name="ys", selection_values=(-1, 1), num_selection_values=100)

    y_small_diff = y_small.to_custom().diff()
    y_small_inv_diff = y_small_inv.to_custom().diff()

    small_coord_system = CoordSystem(d=2, D=4, coord_domain=(-1, 1), coord_func=y_small_inv, inv_func=y_small,
                                     coord_jac=y_small_inv_diff, inv_jac=y_small_diff)

    y = R @ y_small
    y_inv = y.invert(input_name="y", selection_values=(-1, 1), num_selection_values=100)

    y_diff = (R @ y_small.to_custom()).diff()
    y_inv_diff = (y_small_inv.to_custom(Rt @ o)).diff()

    full_coord_system = CoordSystem(d=2, D=48, coord_domain=(-1, 1), coord_func=y_inv, inv_func=y,
                                    coord_jac=y_inv_diff, inv_jac=y_diff)

    return small_coord_system, full_coord_system


def sample_m3():

    # values used for paper experiment
    A = Constant(np.array([[-1.82073578, 0.08264555, 0.23420455],
                           [0.15310137, 1.0151491, -0.14543064],
                           [0.31639373, 0.30489013, 1.39837113]]))
    b = Constant(np.array([0.43044838, -0.61110428, -0.36894417]))

    # A = Constant(np.random.random((3, 3)))
    # b = Constant(np.random.random((3,)))

    R = Constant(special_ortho_group.rvs(48)[:, :7])
    Rt = Constant(R.numpy_value.T)

    x = Var(name="x", shape=(3,))
    o = Var(name="o", shape=(48,))

    h1 = A @ x + b

    h2 = custom(h1, ["1.0/(1.0 + exp(h_0))", "exp(h_1)", "log(1.0 + exp(h_2))"])
    h3 = custom(h2, ["h_1", "log(1.0 + h_0)", "h_0 + h_2", "h_1 + h_2"])
    h4 = custom(h3, ["h_1 * sin(h_3)", "h_0 * cos(h_2)", "cos(2 * h_2)", "sin(2 * h_3)"])

    y_small = concat([h2, h4])
    y_small_inv = y_small.invert(input_name="ys", selection_values=(-1, 1), num_selection_values=100)

    y_small_diff = y_small.to_custom().diff()
    y_small_inv_diff = y_small_inv.to_custom().diff()

    small_coord_system = CoordSystem(d=3, D=7, coord_domain=(-1, 1), coord_func=y_small_inv, inv_func=y_small,
                                     coord_jac=y_small_inv_diff, inv_jac=y_small_diff)

    y = R @ y_small
    y_inv = y.invert(input_name="y", selection_values=(-1, 1), num_selection_values=100)

    y_diff = (R @ y_small.to_custom()).diff()
    y_inv_diff = (y_small_inv.to_custom(Rt @ o)).diff()

    full_coord_system = CoordSystem(d=3, D=48, coord_domain=(-1, 1), coord_func=y_inv, inv_func=y,
                                    coord_jac=y_inv_diff, inv_jac=y_diff)

    return small_coord_system, full_coord_system


def sample_fake_functions(coord_system: CoordSystem, num_fake_functions: int):

    inp_var = coord_system.coord_func.inp_var

    select_true_mat = np.zeros((num_fake_functions, coord_system.d), dtype=float)
    select_true_mat[np.arange(num_fake_functions), np.random.randint(coord_system.d, size=(num_fake_functions, ))] = 1.0
    select_true_mat = Constant(select_true_mat)

    select_inp_mat = np.zeros((num_fake_functions, coord_system.D), dtype=float)
    select_inp_mat[np.arange(num_fake_functions), np.random.randint(coord_system.D, size=(num_fake_functions,))] = 1.0
    select_inp_mat = Constant(select_inp_mat)

    pi = Constant(np.pi)
    alpha = Constant(np.random.uniform(-2.0, 2.0, size=(num_fake_functions, )))

    select_true = (pi * select_true_mat) @ coord_system.coord_func
    fake_funcs = select_inp_mat @ inp_var + alpha * elementwise(select_true, func=sin)

    coord_jac = coord_system.coord_jac.compose(inp_var)
    fake_funcs_jac = pi * reshape(alpha * elementwise(select_true, func=cos), out_shape=(num_fake_functions, 1))
    fake_funcs_jac = select_inp_mat + fake_funcs_jac * select_true_mat @ coord_jac

    return fake_funcs, fake_funcs_jac


# sample a manifold. Only sampling a random affine transformation, the other "parts" of the manifold have to be given.
# The small coord sys refers to the manifold obtained before applying the last rotation matrix.
# We use this for visualizations. The full coord sys refers to the one with R applied.
small_coord_sys, full_coord_sys = sample_m1()

# sample fake functions
fake_coord_func, fake_coord_jac = sample_fake_functions(full_coord_sys, num_fake_functions=36)

# sample data_points
small_coord_sys.generate_data(num_samples=8000)
full_coord_sys.generate_data(num_samples=8000)

# fake data(we only need the gradients of the fake functions)
full_coord_sys.data.data_dict["fake_coord_jac"] = fake_coord_jac(full_coord_sys.data["amb_pts"], batch_input=True)

# sample noisy data
for noise_scale in [0.0, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]:

    data_key = "amb_pts"
    gradients_key = "coord_jac"

    if noise_scale:
        data_key = data_key + "_" + str(noise_scale)
        gradients_key = gradients_key + "_" + str(noise_scale)

    # add gaussian noise, but clip to the bounding box of non-noisy data + some tolerance to not go outside the domain
    # of the inverses
    full_coord_sys.data.add_noise("amb_pts", save_key=data_key, noise_kwargs=dict(scale=noise_scale),
                                  clip_bbox=True, bbox_tol=0.005)

    # evaluate the gradients of the true and fake functions on the noisy data
    full_coord_sys.data.data_dict[gradients_key] = full_coord_sys.coord_jac(full_coord_sys.data[data_key], batch_input=True)
    full_coord_sys.data.data_dict["fake_" + gradients_key] = fake_coord_jac(full_coord_sys.data[data_key], batch_input=True)

    full_coord_sys.data.remove_bad_values()


# check that the data is valid(no-nans, no-infs, funcs and their invs have correct values) and that we have at least
# 5000 values left after removing bad data points
full_coord_sys.data.remove_bad_values()
assert full_coord_sys.data.has_integrity and full_coord_sys.data.num_points >= 5000


# save data relevant for experiments
for noise_scale in [0.0, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]:

    data_key = "amb_pts"
    gradients_key = "coord_jac"

    if noise_scale:
        data_key = data_key + "_" + str(noise_scale)
        gradients_key = gradients_key + "_" + str(noise_scale)

    TSLassoData(data=full_coord_sys.data[data_key][:5000],
                gradients=np.concatenate([full_coord_sys.data[gradients_key],
                                          full_coord_sys.data["fake_" + gradients_key]],
                                         axis=1)[:5000],
                true_idxs=tuple(range(full_coord_sys.d)),
                D=full_coord_sys.D, d=full_coord_sys.d).save_pkl("m1_data_" + str(noise_scale) + "_sigma.pkl")


# visualize the data
small_coord_sys.data.visualize_pts(data_dims=tuple(range(min(4, small_coord_sys.D))), aspect="equal")

# add noise to the small coord sys for visualization.
small_coord_sys.data.add_noise("amb_pts", save_key="amb_pts_0.25", noise_kwargs=dict(scale=0.025))
small_coord_sys.data.add_noise("amb_pts", save_key="amb_pts_0.5", noise_kwargs=dict(scale=0.05))

small_coord_sys.data.visualize_pts(data_key=("amb_pts", "amb_pts_0.25", "amb_pts_0.5"),
                                   data_dims=((0, 1, 2),
                                              tuple(range(small_coord_sys.D, small_coord_sys.D + 3)),
                                              tuple(range(2 * small_coord_sys.D, 2 * small_coord_sys.D + 3))),
                                   aspect='equal')

# print the manifold functions before the rotation matrix
print(small_coord_sys.inv_func.eval_sympy(dict()))

# nu
print("Avg Nu:", full_coord_sys.data.nu(np.mean))
print("Max Nu:", full_coord_sys.data.nu(np.max))

# mu
print("Avg Mu:", full_coord_sys.data.mu("fake_coord_jac", np.mean))
print("Max Mu:", full_coord_sys.data.mu("fake_coord_jac", np.max))

# cond numbers
print("Avg coord jac condition number:",  full_coord_sys.data.cond_num("coord_jac", np.mean))
print("Max coord jac condition number:",  full_coord_sys.data.cond_num("coord_jac", np.max))
print("Avg inv jac condition number:",  full_coord_sys.data.cond_num("inv_jac", np.mean))
print("Max inv jac condition number:",  full_coord_sys.data.cond_num("inv_jac", np.max))
