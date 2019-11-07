import matplotlib.pyplot as plt
import numpy as np

from src.randgg.algorithm import compute_closest_intersections, compute_regressions


def plot_curve(data_frame, nb_samples, title, xlim = (0.0, 1.0), savefile = None):
    mid = nb_samples / 2
    step = nb_samples / 10
    ylim = (0, nb_samples)

    plt.figure()

    _ = data_frame.plot(
        x = 'p',
        figsize = (20, 10),
        grid = True,
        xlim = xlim,
        ylim = ylim,
        yticks = np.arange(0, nb_samples, step = step),
    )

    closest_intersection = compute_closest_intersections(data_frame)

    for k, (v, p) in closest_intersection.items():
        plt.plot(p, [mid], 'ro')

    # plot the line where 50% are with the property and 50% are not
    plt.axhline(y = mid, color = 'r')

    plt.title(title)
    plt.legend(list([str(k) + ' : ' + str(p) for (k, (v, p)) in closest_intersection.items()]))
    plt.ylabel('number of graph satisfying the property')
    plt.xlabel('$p$')

    if savefile is not None:
        plt.savefig(savefile, bbox_inches = 'tight')
    plt.show()


def plot_regressions(df, col, nb_samples, title,
                     min_degree = 1,
                     max_degree = 12,
                     step_degree = 2,
                     subplot = (3, 4),
                     savefile = None,
                     ):
    mid = nb_samples / 2
    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    x = df['p'].values
    y = df[col].values

    regs = compute_regressions(df, col, max_degree, nb_samples)

    plt.figure(figsize = (40, 20))

    d = min_degree
    start = 1
    while d <= max_degree:
        plt.subplot(subplot[0], subplot[1], start)
        plt.title('linear regression of degree ' + str(d) + ', error (RMSE) : ' + str(
            regs[d][2]) + '\n' + 'value of p for 50% : ' + str(regs[d][3]))
        fit_fn = regs[d][1]
        plt.xlim((0.0, 1.0))
        plt.ylim((0, nb_samples))
        plt.scatter(x, y)
        plt.ylabel('number of graph satisfying the property')
        plt.xlabel('$p$')
        plt.plot(x, y, color[d % 7] + 'o', x, fit_fn(x), '--' + color[(d + 6) % 7])
        plt.axhline(y = mid, color = 'r')
        plt.legend(['samples', 'regression curve', '50%'])
        d = d + step_degree
        start = start + 1

    if savefile is not None:
        plt.savefig(savefile, bbox_inches = 'tight')

    plt.title(title)
    plt.show()
