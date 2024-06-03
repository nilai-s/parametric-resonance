from src import axion
from src import correlator

import numpy
import pickle
import matplotlib.pyplot as plt

import os
import argparse
import pathlib
import pprint
import tomllib

def plot_axions(experiment):
    '''
    Generate sigma values for a range of axion parameters
    Plot the axion field strength.
    '''
    plt.figure()
    exp_dir = pathlib.Path('experiments/%s/' % experiment)
    in_dir  = exp_dir / "in"
    out_dir = exp_dir / "out"
    configurations = [os.path.splitext(fn)[0] for fn in os.listdir(in_dir)]
    for cfg_name in configurations:
        cfg_path = in_dir / ("%s.toml" % cfg_name)
        assert cfg_path.is_file()

        with open(cfg_path, 'rb') as handle:
            config = tomllib.load(handle)
        pprint.pprint(config)

        A = axion.Axion(config)
        plt.plot(A.t, A.s_re * numpy.exp(-A.t), 'x--', label='%s,%s' % (cfg_name, repr(A)))

    plt.title('Axion mode function, experiment=%s' % experiment)
    plt.xlabel('x')
    plt.ylabel('$\\sigma_+(x)$')
    plt.legend()
    plt.grid(which='both')
    plt.savefig(out_dir / 'axion_field.png', dpi=480)
    plt.close()

def plot_correlator(experiment):
    '''
    Plot correlator for a range of u values.
    '''
    exp_dir = pathlib.Path('experiments/%s/' % experiment)
    in_dir  = exp_dir / "in"
    out_dir = exp_dir / "out"
    with open(out_dir / ('data.pickle'), 'rb') as handle:
        output = pickle.load(handle)
    plt.figure()
    plt.title('++ component of 4pcf, experiment=%s' % experiment)
    for name, result in output.items():
        u, F = result
        plt.plot(u, F, 'x--', label=name)
    plt.xscale('log')
    plt.xlabel('u')
    plt.ylabel('$Re(\\tilde{F}(x, x))$')
    plt.legend()
    plt.grid(which='major')
    plt.savefig(out_dir / 'F_tilde.png' , dpi=480)
    plt.close()

def generate_correlator(experiment):
    '''
    Generate \tilde{F} values for a range of axion parameters
    '''
    output  = {}
    exp_dir = pathlib.Path('experiments/%s/' % experiment)
    in_dir  = exp_dir / "in"
    out_dir = exp_dir / "out"
    configurations = [os.path.splitext(fn)[0] for fn in os.listdir(in_dir)]
    for cfg_name in configurations:
        cfg_path = in_dir / ("%s.toml" % cfg_name)
        assert cfg_path.is_file()

        with open(cfg_path, 'rb') as handle:
            config = tomllib.load(handle)
        pprint.pprint(config)

        C = correlator.Correlator(config)
        u, F = C.integrate()
        output[cfg_name] = (u, F)

    with open(out_dir / "data.pickle", 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-D', '--experiment_dir', dest='exp_dir', type=str,
                        help='The name (output directory) for this experiment.')
    parser.add_argument('mode', choices=['setup', 'run', 'plot'])
    args   = parser.parse_args()
    if args.mode == 'setup':
        exp_dir = pathlib.Path('experiments/%s/' % args.exp_dir)
        if exp_dir.is_dir():
            print('NOTE: Experiment directory %s exists already; results will be overwritten!' % exp_dir)
        in_dir  = exp_dir / "in"
        out_dir = exp_dir / "out"
        in_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        print('Please propagate config files into the directory at %s' % in_dir)
    elif args.mode == 'run':
        generate_correlator(args.exp_dir)
        # plot_axions(args.exp_dir)
        # plot_correlator(args.exp_dir)
    elif args.mode == 'plot':
        plot_axions(args.exp_dir)
        plot_correlator(args.exp_dir)
