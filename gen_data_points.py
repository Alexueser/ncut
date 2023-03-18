import numpy as np


def make_data_fixed_points():
    return np.array([(1, 1), (1, 2), (2, 1), (2, 2),
                     (1, 3), (1, 4), (2, 3), (2, 4),
                     (3, 1), (3, 2), (4, 1), (4, 2),
                     (10, 1), (10, 2), (1, 10), (2, 10)])


def make_data_two_part():
    sigma_h = 2
    sigma_v = 10

    s_v = 10
    s_h = 30

    a = np.array([sigma_h * np.random.random(40),
                  sigma_v * np.random.random(40)])
    b = np.array([sigma_h * np.random.random(50),
                  sigma_v * np.random.random(50)]) + \
        np.array([[s_h], [s_v]]) * np.ones((1, 50))
    return np.hstack((a, b)).transpose()


def make_data_three_part():
    radius = 15
    size_cluster = [80, 20, 20]
    raw_data = np.random.randn(2, sum(size_cluster))
    tmp = np.random.randn(2, size_cluster[0]) - 0.5
    idt = tmp[1].argsort()

    r_noise = 4
    raw_data2 = np.array([tmp[0, idt] * r_noise,
                          tmp[1, idt] * 2])

    data1 = np.array([(radius - raw_data2[0, 0:size_cluster[0]]) * np.cos(np.pi * raw_data2[1, 0:size_cluster[0]]),
                      (radius - raw_data2[0, 0:size_cluster[0]]) * np.sin(np.pi * raw_data2[1, 0:size_cluster[0]])])

    center = np.array([0, 0])
    sig = np.array([1, 2])

    scb = size_cluster[0] + 1
    scb_next = scb + size_cluster[1] - 1
    data2 = np.array([center[0] + sig[0] * raw_data[0, scb:scb_next],
                      center[1] + sig[1] * raw_data[1, scb:scb_next]])

    center = np.array([radius + 20, 0])
    sig = np.array([1, 1])
    scb = scb_next + 1
    scb_next = scb + size_cluster[2] - 1
    data3 = np.array([center[0] + sig[0] * raw_data[0, scb:scb_next],
                      center[1] + sig[1] * raw_data[1, scb:scb_next]])

    return np.hstack((data1, data2, data3)).transpose()


def make_points(points_type):
    data = {1: make_data_fixed_points,
            2: make_data_two_part,
            3: make_data_three_part}
    return data[points_type]()