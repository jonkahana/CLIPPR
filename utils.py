

from os.path import join
import os


class DictX(dict):
    """
    Taken From https://dev.to/0xbf/use-dot-syntax-to-access-dictionary-key-python-tips-10ec
    """

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __repr__(self):
        return '<DictX ' + dict.__repr__(self) + '>'


def save_experiment_hyper_params(args, exp_dir, verbose=True):
    with open(join(exp_dir, f'args.txt'), 'w+') as f:
        # print args to file
        f.write('\n\n\n')
        f.write('Experiment Args:\n\n')
        for k in args:
            f.write(f'\t {k}: {args[k]}\n')  # print config to file
        f.write('\n\n\n')
    if verbose:
        with open(join(exp_dir, f'args.txt'), 'r') as f:
            for line in f:
                print(line)
    return
