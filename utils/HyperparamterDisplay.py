from __future__ import print_function, absolute_import


def display(args):
    #  Display information of current training
    print('Learn Rate  \t%.1e' % args.lr)
    print('Epochs  \t%05d' % args.epochs)
    print('Log Path \t%s' % args.log_dir)
    print('Data Set \t %s' % args.data)
    print('Batch Size  \t %d' % args.BatchSize)
    print('Embedded Dimension \t %d' % args.num_class)
    print('Begin to train the network')
    print(50 * '#')
