import numpy as np
import pyjet

def subjet_axes(event, R):
    # convert array to numpy structured array
    if event.shape[-1] ==5:
        particles = np.array([tuple(i) for i in event], dtype=[('pt','f8'), ('eta','f8'), ('phi','f8'), ('mass','f8'), ('pid','f8')])
    else:
        particles = np.array([tuple(i) for i in event], dtype=[('pt','f8'), ('eta','f8'), ('phi','f8'), ('mass','f8')])
    # cluster the subjets
    sequence = pyjet.cluster(particles, algo='kt', R=R)
    subjets = sequence.inclusive_jets()
    # ensure we have physical subjets
    subjets = list(filter(lambda x: x.eta < 4, subjets))
    # define the subjet axes
    axes = np.asarray([[i.eta, i.phi] for i in subjets])
    return axes

def rotate_points(xs, ys, angle):
    x_new = np.cos(angle)*xs - np.sin(angle)*ys
    y_new = np.cos(angle)*ys + np.sin(angle)*xs
    return x_new, y_new

def reflect_event(event, axes, end):
    try:
        if axes[2,1] < 0: event[:end, 2] *= -1
    except IndexError:
        weighted_phi = sum([event[i,0]*event[i,2] for i in range(end)])
        if weighted_phi < 0: event[:end,2] *= -1

def rotate_event(event, axes, end):
    try:
        angle = np.arctan2(axes[1][0] - axes[0][0], axes[1][1]-axes[0][1])
    except IndexError:
        return 1
    theta = -0.5*np.pi - angle
    event[:end,2], event[:end,1] = rotate_points(event[:end,2], event[:end,1], theta)
    try:
        axes[2,1], axes[2,0] = rotate_points(axes[2,1], axes[2,0], theta)
    except IndexError:
        return 1

def center_event(event, axes, end):
    event[:end,1:3] -= axes[0]
    event[:end,2] = np.mod(event[:end,2] + np.pi, 2*np.pi) - np.pi
    axes -= axes[0]
    axes[:,1] = np.mod(axes[:,1] + np.pi, 2*np.pi) - np.pi

def preprocess(data, R, norm=True, rotate=True, reflect=True):
    for event in data:
        # find beginning of zero padding
        end = np.count_nonzero(event[:,0])
        # find subjet axes
        axes = subjet_axes(event[:end], R)
        # normalise pT
        if norm: event[:end,0] /= event[:end,0].sum()
        # center, rotate and reflect event
        center_event(event, axes, end)
        if rotate: rotate_event(event, axes, end)
        if reflect: reflect_event(event, axes, end)

def weight_from_val(val, hist, edges):
    index = np.count_nonzero(val>edges) - 1
    return 1/hist[index]

def weights_from_obs(obs, bins=100):
    hist, edges = np.histogram(obs, bins=bins, density=True)
    return np.array(list(map(lambda x: weight_from_val(x, hist, edges), obs)))