"""

"""


import numpy as np
import random
from itertools import izip, cycle, repeat, count
import json
import os

from gnumpy import garray
from gnumpy import max as gmax
from gnumpy import sum as gsum
from gnumpy import newaxis as gnewaxis
from gnumpy import exp as gexp
from gnumpy import log as glog
import chopmunk as munk
import climin.util


def _cycle(data, btsz):
    """
    """
    bgn = cycle(xrange(0, data.shape[0]-btsz+1, btsz))
    end = cycle(xrange(btsz, data.shape[0]+1, btsz))
    return bgn, end


def cycle_inpt(inputs, btsz, **kwargs):
    """
    """
    bgn, end = _cycle(inputs, btsz)
    for idx, idx_p1 in izip(bgn, end):
        yield garray(inputs[idx:idx_p1])


def cycle_noisy_inpt(inputs, btsz, noise, **kwargs):
    """
    """
    bgn, end = _cycle(inputs, btsz)
    for idx, idx_p1 in izip(bgn, end):
        _inputs = inputs[idx:idx_p1]
        noisify = np.random.rand(*_inputs.shape) > noise
        noisify = noisify * _inputs
        yield garray(noisify)


def cycle_gaussian_inpt(inputs, btsz, std, **kwargs):
    """
    """
    bgn, end = _cycle(inputs, btsz)
    for idx, idx_p1 in izip(bgn, end):
        _inputs = inputs[idx:idx_p1]
        noisify = np.random.randn(*_inputs.shape)
        noisify *= _inputs
        yield garray(noisify)


def cycle_trgt(targets, btsz, **kwargs):
    """
    """
    bgn, end = _cycle(targets, btsz)
    for idx, idx_p1 in izip(bgn, end):
        yield garray(targets[idx:idx_p1])


def cycle_pairs(inputs, btsz, **kwargs):
    """
    """
    p0, p1 = inputs[0], inputs[1]
    bg, end = _cycle(p0, btsz)
    for idx, idx_p1 in izip(bg, end):
        yield (garray(p0[idx:idx_p1]), garray(p1[idx:idx_p1]))


def jump(frm, to, when):
    i = 0
    while True:
        if i >= when:
            yield to
        else:
            yield frm
            i = i+1


def lin_inc(frm, to, step, end):
    i = 0
    diff = to - frm
    delta = end/(1.0*step)
    inc = diff/delta
    # minus inc handels divmod/i=0 case.
    strt = frm - inc
    while True:
        if i >= end:
            yield to
        else:
            d, r = divmod(i, step)
            if r == 0:
                strt += inc
            yield strt
            i = i + 1


def const(const):
    while True:
        yield const


def momentum_schedule(max_momentum):
    while True:
        m = 1 - (2 ** (-1 - np.log(np.floor_divide(i, 50) + 1, 2)))
        yield min(m, max_momentum)


def two_step(step_one, step_two):
    for s1, s2 in izip(step_one, step_two):
        yield (s1, s2)


def range_inpt(inputs, btsz, **kwargs):
    return lambda idx: garray(inputs[idx:idx+btsz])


def range_trgt(targets, btsz, **kwargs):
    return lambda idx: garray(targets[idx:idx+btsz])


def range_noisy_inpt(inputs, btsz, noise, **kwargs):
    def noisify(idx):
        _inputs = inputs[idx:idx+btsz]
        noisify = np.random.rand(*_inputs.shape) > noise
        noisify = noisify * _inputs
        return garray(noisify)
    return noisify


def range_pairs(inputs, btsz, **kwargs):
    return lambda idx: (garray(inputs[0][idx:idx+btsz]), garray(inputs[1][idx:idx+btsz]))


external_iargs = {
    cycle_inpt: {"inputs": "inputs"}
    ,cycle_noisy_inpt: {"inputs": "inputs", "noise": "noise"}
    ,cycle_trgt: {"targets": "targets"}
    ,cycle_pairs: {"inputs": "inputs"}
}


finite_arg = {
    cycle_inpt: range_inpt
    ,cycle_noisy_inpt: range_noisy_inpt
    ,cycle_trgt: range_trgt
    ,cycle_pairs: range_pairs
}


def logsumexp(array, axis=0):
    """
    Compute log of (sum of exps)
    along _axis_ in _array_ in a
    stable way.
    """
    axis_max = gmax(array, axis)[:, gnewaxis]
    return axis_max + glog(gsum(gexp(array-axis_max), axis))[:, gnewaxis]


def _logsumexp(array, axis=0):
    """
    """
    axis_max = np.max(array, axis)[:, np.newaxis]
    return axis_max + np.log(np.sum(np.exp(array-axis_max), axis))[:, np.newaxis]


def prepare_opt(opt_schedule, wrt, schedule, train, valid, test=None):
    # iargs, a generator passed to climin optimizer,
    # is build out of generators on the fly -- needs to know what
    # parameters those generators must be called with.
    opt_schedule["inputs"] = train[0]
    opt_schedule["targets"] = train[1]

    iargs=[]
    for arg in opt_schedule["iargs"]:
        needed_args = external_iargs[arg]
        for n in needed_args:
            # get only arguments that are not yet available
            if n not in opt_schedule:
                opt_schedule[n] = schedule[needed_args[n]]
        iargs.append(arg(**opt_schedule))
    iargs = izip(*iargs)

    ikwargs = repeat({})

    opt_schedule["train"] = train
    opt_schedule["valid"] = valid
    if test is not None:
        opt_schedule["test"] = test
    if "eval" not in opt_schedule:
        opt_schedule["eval"] = schedule["eval"]

    evals, peeks = eval_opt(opt_schedule)

    opt_keys = opt_schedule.keys()
    for arg in opt_schedule["iargs"]:
        needed_args = external_iargs[arg]
        for n in needed_args:
            if n in opt_schedule and n not in opt_keys:
                del opt_schedule[n]
    # get optimizer
    opt = opt_schedule["type"]
    opt_schedule["args"] = izip(iargs, ikwargs)
    opt = climin.util.optimizer(opt, wrt, **opt_schedule)
    return opt, evals, peeks


def eval_opt(schedule):
    btsz = schedule["btsz"]
    scores = [schedule["f"]]
    if "eval_score" in schedule:
        scores.append(schedule["eval_score"])

    evals = {}
    for e in schedule["eval"]:
        args = []
        schedule["inputs"] = schedule[e][0]
        schedule["targets"] = schedule[e][1]
        for arg in schedule["iargs"]:
            args.append(finite_arg[arg](**schedule))
        inputs = schedule["inputs"]

        def loss(wrt, inputs=inputs, args=args):
            acc = [0] * len(scores)
            if type(inputs) is tuple:
                N = inputs[0].shape[0]
            else:
                N = inputs.shape[0]
            for idx in xrange(0, N - btsz + 1, btsz):
                for j, score in enumerate(scores):
                    acc[j] += score(wrt, *[arg(idx) for arg in args])
            return acc

        evals[e] = loss

    peeks = {}
    if "peeks" in schedule:
        N = schedule["peek_samples"]
        tmp = schedule["btsz"]
        schedule["btsz"] = N
        for p in schedule["peeks"]:
            args = []
            schedule["inputs"] = schedule[p][0]
            schedule["targets"] = schedule[p][1]
            for arg in schedule["iargs"]:
                args.append(finite_arg[arg](**schedule))
            inputs = schedule["inputs"]
            def peek(wrt, inputs=inputs, args=args):
                samples = scores[0](wrt, *[arg(0) for arg in args], predict=True)
                return samples, inputs[:N]
        peeks[p] = peek
        schedule["btsz"] = tmp

    return evals, peeks


def replace_gnumpy_data(item):
    if isinstance(item, dict):
        item = dict((k, replace_gnumpy_data(item[k])) for k in item)
    elif isinstance(item, list):
        item = [replace_gnumpy_data(i) for i in item]
    elif isinstance(item, tuple):
        item = tuple(replace_gnumpy_data(i) for i in item)
    elif isinstance(item, garray):
        if item.size > 1:
            item = item.abs().mean()
    return item


def load_params(fname):
    d = dict()
    with open(fname) as f:
        for line in f:
            tmp = json.loads(line)
            tmp["params"] = np.asarray(tmp["params"], dtype=np.float32)
            d[tmp["layer"]] = tmp
    return d


def load_sched(depot, folder, tag):
    """
    depot, abs path
    """
    import cPickle
    # import fnmatch
    # fdir = os.path.join(depot, folder, tag)
    # l_fname = fnmatch.filter(os.listdir(fdir), '*.schedule')
    fname = os.path.join(depot, folder, tag + '.schedule')
    sched_f = open(fname)
    sched = cPickle.load(sched_f)
    sched_f.close()
    return sched


def log_queue(log_to=None):
    if log_to:
        # standard logfile
        jlog = munk.file_sink(log_to+".log")
        jlog = munk.jsonify(jlog)
        jlog = munk.timify(jlog, tag="timestamp")
        jlog = munk.exclude(jlog, "params")

        # parameter logfile
        paraml = munk.file_sink(log_to+".params")
        paraml = munk.jsonify(paraml)
        paraml = munk.timify(paraml, tag="timestamp")
        paraml = munk.include(paraml, "params")

        jplog = munk.broadcast(*[jlog, paraml])

        # finally a pretty printer for some immediate feedback
        pp = munk.timify(munk.prettyprint_sink())
        pp = munk.dontkeep(pp, "tags")
        pp = munk.include_tags_only(pp, "pretty")

        jplog = munk.exclude_tags(jplog, "pretty")

        log = munk.broadcast(*[jplog, pp])
    else:
        pp = munk.timify(munk.prettyprint_sink())
        pp = munk.dontkeep(pp, "tags")
        log = munk.include_tags_only(pp, "pretty")
    return log


def reload(depot, folder, tag, layer):
    """
    """
    import notebook as nb
    model, schedule = nb.reload(depot, folder, tag, layer)

    log = munk.prettyprint_sink()
    log = munk.dontkeep(log, "tags")
    log = munk.include_tags_only(log, "pretty")

    schedule['logging'] = log

    lab = schedule['__lab__']
    lab = __import__(lab.split('.')[0])
    lab.no_training(model, schedule)


def init_SI(shape, sparsity):
    """
    Produce sparsely initialized weight matrix
    as described by Martens, 2010.

    Note: shape is supposed to be visible x hiddens.
    The following code produces first a hiddens x visible.
    """
    tmp = np.zeros((shape[1], shape[0]))
    for i in tmp:
        i[random.sample(xrange(shape[0]), sparsity)] = np.random.randn(sparsity)
    return tmp.T


def binomial(width):
    filt = np.array([0.5, 0.5])
    for i in xrange(width-2):
        filt = np.convolve(filt, [0.5, 0.5])
    return filt


def mask(factors, stride, size):
    fsqr = int(np.sqrt(factors))
    hsqr = int(fsqr/stride)
    conv = np.zeros((factors, hsqr*hsqr), dtype=np.float32)
    msk = np.zeros((factors, hsqr*hsqr), dtype=np.float32)
    _s = size/2
    print "Mask size:", msk.shape
    col = np.zeros((1, fsqr))
    col[0, 0:size] = binomial(size)
    row = np.zeros((1, fsqr))
    row[0, 0:size] = binomial(size)
    for j in xrange(0, fsqr, stride):
        for i in xrange(0, fsqr, stride):
            _row = np.roll(row, j-_s)
            _col = np.roll(col, i-_s)
            idx = (j*hsqr + i)/stride
            conv[:, idx] = np.dot(_col.T, _row).ravel()
            msk[:, idx] = conv[:, idx] > 0
    return msk, conv


def reload_checker(schedule, mode, l=None):
    '''
    check the must-match schedule
    mode, 'layer' or 'stack'
    l,  which layer to check and load
    return reloaded file name and parameters
    '''
    assert (mode == 'layer' or mode == 'stack'), \
        'wrong mode selection in reloading'
    from os.path import join
    depot = schedule['config_reload']['depot']
    folder = schedule['config_reload']['folder']
    tag = schedule['config_reload']['tag']
    reload_schedule = load_sched(depot, folder, tag)
    if 'data_file' in reload_schedule:
        assert (schedule['data_file'] == reload_schedule['data_file']), \
            'data file must be identical with current one'
    if mode == 'stack':
        rs = reload_schedule.copy()
        s = schedule.copy()
    else:
        rs = reload_schedule['stack'][l].copy()
        s = schedule['stack'][l].copy()
    reload_epochs = rs['opt']['epochs']
    assert (rs['opt']['epochs'] <= s['opt']['epochs']), \
        'epochs of reload schedule must not larger than current one'
    rs['opt'].pop('epochs')
    s['opt'].pop('epochs')
    if 'block' in rs:
        rs.pop('block')
    if 'block' in s:
        s.pop('block')
    if 'rho' not in rs and 'rho' in s:
        assert (s['rho'] == 0), 'rho option mismatch in reloading'
        s.pop('rho')
    if 'lmbd' not in rs and 'lmbd' in s:
        assert (s['lmbd'] == 0), 'lmbd option mismatch in reloading'
        s.pop('lmbd')
    if 'dropout' not in rs and 'dropout' in s:
        assert (s['dropout'] == None), 'dropout option mismatch in reloading'
        s.pop('dropout')
    # print schedule['opt']['epochs']
    # print reload_schedule['opt']['epochs']
    # rs['opt'].pop('momentum')
    # s['opt'].pop('momentum')
    # s_stop = s['opt']['stop']
    # rs['opt'].pop('stop')
    # s['opt'].pop('stop')
    if 'auto_weight' not in rs and 'auto_weight' in s:
        assert (r['auto_weight'] == False), 'auto weight option mismatch in reloading'
    if 'class_weight' in s:
        s.pop('class_weight')
    if mode == 'stack':
        assert (rs['opt'] == s['opt']),'opt mis'
        assert (rs['score'] == s['score']),'score mis'
        # assert (rs['opt'] == s['opt'] and rs['stack'] == s['stack']
        #         and rs['score'] == s['score']), \
        'reload schedule must be identical with current one'
    else:
        # print 'rs', rs
        # print 's', s
        assert (rs == s), \
            'reload schedule must be identical with current one'

    # if mode == 'stack':
    #     schedule['opt']['stop'] = int(s_stop)
    # else:
    #     schedule['stack'][l]['opt']['stop'] = int(s_stop)

    fname = join(depot, folder, tag + ".params")
    params = load_params(fname)
    return fname, params, reload_epochs

def epoch_checker(schedule, epoch_index):
    '''
    check the epoch match in previous run
    epoch_index, index of current index in stack(l1, l2...or stack)
    e.g., if epoch_index == stack, than all the epochs in pretraining must be identical
    '''
    depot = schedule['config_reload']['depot']
    folder = schedule['config_reload']['folder']
    tag = schedule['config_reload']['tag']
    reload_schedule = load_sched(depot, folder, tag)
    rs = reload_schedule['stack'][:epoch_index]
    s = schedule['stack'][:epoch_index]
    for i in range(epoch_index):
        assert (rs[i]['opt']['epochs'] == s[i]['opt']['epochs']), \
            'epochs must match in previous run!'
    return 0


def reload_info(schedule, mode, l=None):
    '''
    information of reloaded schedule
    mode, 'layer' or 'stack'
    l,  which layer to check and load
    return reloaded file name and reload_epochs, reload_stop
    '''
    assert (mode == 'layer' or mode == 'stack'), \
        'wrong mode selection in reloading'
    from os.path import join
    depot = schedule['config_reload']['depot']
    folder = schedule['config_reload']['folder']
    tag = schedule['config_reload']['tag']
    reload_schedule = load_sched(depot, folder, tag)
    if mode == 'stack':
        rs = reload_schedule
    else:
        rs = reload_schedule['stack'][l]
    reload_epochs = rs['opt']['epochs']
    reload_stop = rs['opt']['stop']
    fname = join(depot, folder, tag + ".log")
    return fname, reload_epochs, reload_stop

def reload_log(log, schedule, mode):
    '''
    reload the log file
    '''
    assert (mode == 'layer' or mode == 'stack'), \
        'wrong mode selection in reloading'
    line_num = 0
    if mode == 'stack':
        load_range = len(schedule['stack']) - 1
    if mode == 'layer':
        load_range = schedule["pretrain_before"]
    for i in range(load_range):
        fname, reload_epochs, reload_stop = reload_info(schedule, 'layer', i)
        line_num += reload_epochs / reload_stop
    if mode == 'stack':
        fname, reload_epochs, reload_stop = reload_info(schedule, 'stack')
        line_num += reload_epochs / reload_stop
    load_log_write(log, fname, line_num)
    return fname


def load_log_write(log, fname, l):
    """
    l, line number
    load the .log file from 'fname'
    then send to the consumer
    """
    import ast
    c = 0
    with open(fname, 'r') as in_f:
        for line in in_f:
            if c < l:
                line_dict = ast.literal_eval(line)
                log.send(line_dict)
            c += 1


def plot_log(depot, folder, tag):
    '''
    visualize log file
    n, total number of samples
    '''
    import ast
    import math as m
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'font.size': 8})


    fname = os.path.join(depot, folder, tag + '.log')
    image = os.path.join(depot, folder, tag + '.png')
    data = []
    layer = None
    with open(fname, 'r') as f:
        for line in f:
            line_dict = ast.literal_eval(line)
            n_layer = line_dict['layer']
            if n_layer != layer:
                data.append({})
                data[-1].update({'layer': n_layer})
                data[-1].update({'train': []})
                data[-1].update({'valid': []})
                data[-1].update({'gradient': []})
            if type(line_dict['train'][0]) is not list:
                data[-1]['train'].append(line_dict['train'][0])
                data[-1]['valid'].append(line_dict['valid'][0])
                data[-1]['gradient'].append(line_dict['gradient'])
            else:
                data[-1]['train'].append(line_dict['train'][0][0])
                data[-1]['valid'].append(line_dict['valid'][0][0])
                data[-1]['gradient'].append(line_dict['gradient'])
            layer = n_layer
    subplot_shape = (m.ceil(len(data)/2.), 2)
    fig = plt.figure()
    k = 1
    for l in data:
        f = fig.add_subplot(subplot_shape[0], subplot_shape[1], k)
        f.set_ylabel('error')
        f.set_xlabel('error')
        f.set_title('{} layer'.format(l['layer']))
        train_error = l['train']
        valid_error = l['valid']
        num = len(train_error)
        f.plot(range(num), train_error, color='blue')
        f.plot(range(num), valid_error, color='red')
        k += 1

    fig.savefig(image, dpi=160)
    # fig.savefig(os.path.join(
    #     image_dir, '{}-{}-{}-{}.png'.format(
    #         layer, begin_, end_, n_feature)), format='png',
    #         figsize=(8, 6), dpi=160, facecolor='w',
    #         edgecolor='k')
    plt.clf()
    plt.close(fig)
    return 0

def perf_queue(log_to):
    '''
    performance queue
    '''
    # accuracy file
    acc_log = munk.file_sink(log_to+".accuracy")
    acc_log = munk.jsonify(acc_log)
    acc_log = munk.timify(acc_log, tag="timestamp")
    acc_log = munk.exclude(acc_log, "precision")
    acc_log = munk.exclude(acc_log, "recall")

    # precision file
    prec_log = munk.file_sink(log_to+".precision")
    prec_log = munk.jsonify(prec_log)
    prec_log = munk.timify(prec_log, tag="timestamp")
    prec_log = munk.exclude(prec_log, "recall")

    # recall file
    re_log = munk.file_sink(log_to+".recall")
    re_log = munk.jsonify(re_log)
    re_log = munk.timify(re_log, tag="timestamp")
    re_log = munk.include(re_log, "recall")

    log = munk.broadcast(*[acc_log, prec_log, re_log])

    return log


def visualize_filter(depot, folder, tag, layer, dim, shape_r=None, xtiles=None,
                     fill=0, unblock=None, xs=None, block=None):
    '''
    _layer_ index, start from 0, end with 'Stack' if needed
    _dim_, the input dimension
    _dim_ must be a square number
    if _shape_r is None.
    Specifiy the number of rows with _xtiles_.
    If not specified, the layout is approximately
    square. _fill_ defines the pixel border between
    patches (default is black (==0)).
    '''
    from os.path import join
    from helpers import helpers
    im_dir = join(depot, folder, tag + '.plot')
    if not os.path.isdir(im_dir):
        os.makedirs(im_dir)

    params = load_params(join(depot, folder, tag + '.params'))

    g = lambda x : isinstance(x, int)
    l = filter(g, params.keys())
    max_layer = max(l)

    filters = np.eye(dim)
    if g(layer):
        im_f = join(im_dir, 'layer_{}'.format(layer))
        for i in range(layer + 1):
            shape = params[i]['shape']
            m_end = shape[0] * shape[1]
            filters = np.dot(filters, params[i]['params'][:m_end].reshape(shape))
        im_f += '_unit_{}_filter.png'.format(filters.shape[1])
    if layer is 'Stack':
        im_f = join(im_dir, 'stack')
        current_index = 0
        for i in range(max_layer + 1):
            shape = params[i]['shape']
            m_end = shape[0] * shape[1]
            l = m_end + shape[1]
            filters = np.dot(filters, params['Stack']['params'][current_index:current_index+m_end].reshape(shape))
            current_index += l
        im_f += '_unit_{}_filter.png'.format(filters.shape[1])
    if layer is 'fine_tuning':
        current_index = 0
        for i in range(max_layer + 1):
            im_f = join(im_dir, 'ft_layer_{}'.format(i))
            shape = params[i]['shape']
            m_end = shape[0] * shape[1]
            l = m_end + shape[1]
            filters = np.dot(filters, params['Stack']['params'][current_index:current_index+m_end].reshape(shape))
            im_f += '_unit_{}_filter.png'.format(filters.shape[1])
            current_index += l
            pl_filters = filters.T
            if unblock:
                pl_filters = helpers._batch_unblock_view(pl_filters, xs, block)
            im = helpers.visualize(pl_filters.reshape(-1), dim, shape_r=shape_r, xtiles=xtiles, fill=fill)
            im.save(im_f)
        return None
    if layer is 'all':
        layer_list = range(max_layer + 1)
        layer_list.extend(['Stack','fine_tuning'])
        print layer_list
        for ll in layer_list:
            print 'plotting', ll, 'filter'
            visualize_filter(depot, folder, tag, ll, dim, shape_r, xtiles, fill, unblock, xs, block)
        return None
    filters = filters.T
    if unblock:
        filters = helpers._batch_unblock_view(filters, xs, block)
    im = helpers.visualize(filters.reshape(-1), dim, shape_r=shape_r, xtiles=xtiles, fill=fill)
    im.save(im_f)
