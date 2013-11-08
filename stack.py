"""
YADAYADA is a simple toolbox for training neural networks with many hidden layers.
It allows to build these deep networks by successivly pretraining these layers in
order to help training the complete deep network. Therefore, the central data structure
is the Stack, a list of layers.
Two aspects are at the center of YADAYADA: (i) It must allow pretraining the single
layers and (ii) it must allow the usage of general optimization algorithms. To facilitate
this, the Stack is managing the complete parameter vector of the deep network. The
various layers only have views into this vector -- they have their own parameters when
in pretraining mode. It only uses its view on the global parameter vector when updating
this global parameter after a finished pretraining session. This explains why every single layer gets always the parameters
passed in when doing forward or backward computations over the full stack.
"""


import numpy as np
from itertools import izip
import h5py
from time import strftime


from gnumpy import zeros as gzeros
import gnumpy as gpu

from losses import loss_table
from utils import prepare_opt, replace_gnumpy_data, epoch_checker, reload_checker, reload_log
import chopmunk as munk


class Stack(list):
    def __init__(self, ind, schedule):
        gpu.seed_rand(seed=None)
        self.logging = schedule["logging"]
        # self.performance = shedule["performance"]
        self.psize = 0
        cuts = [0]
        self.stack = schedule["stack"]
        for layer in self.stack:
            ltype = layer["type"]
            units = layer["units"]
            l = ltype.__new__(ltype)
            l.__init__(shape=(ind, units), **layer)
            self.psize += l.size
            self.append(l)
            cuts.append(l.size)
            ind = units
        self.params = gzeros(self.psize)
        self.cuts = np.cumsum(cuts)
        for layer, (c1, c2) in izip(self, izip(self.cuts[:-1], self.cuts[1:])):
            layer.p = self.params[c1:c2]
        if "score" in schedule:
            self._score = schedule["score"]
        else:
            print("You may have a problem: _score_ is NONE")
            self._score = None

        # if len(schedule["train"]) == 1:
        #     self.train_data = [schedule["train"][0]]
        # else:
        #     self.train_data = (None if not schedule.get("train")
        #                        else [schedule["train"][0], schedule["train"][1]])
        self.train_data = (None if not schedule.get("train")
                           else [schedule["train"][0], schedule["train"][1]])
        self.valid_data = (None if not schedule.get("valid")
                           else [schedule["valid"][0], schedule["valid"][1]])
        self.test_data = (None if not schedule.get("test")
                          else [schedule["test"][0], schedule["test"][1]])
    def __repr__(self):
        rep = "|".join([str(l) for l in self])
        return rep

    def pretrain(self, schedule):
        # make sure reload the pretrain parameters and reload the train parameters won't happen together
        assert (not (schedule["reload_pretrain"] & schedule["reload_train"])), \
            "reload pretrain and reload_train mismatch"

        if schedule["reload_train"]:
            for i, (layer, sched) in enumerate(izip(self, self.stack)):
                log = munk.add_keyvalue(self.logging, "layer", i)
                pt_params = layer.pt_init(**sched)
                pt_params, _ = self.reload_one_layer(schedule, i)
                info = layer.pt_done(pt_params, **sched)
                pt_params = None
                log.send(info)
            pp = {"msg": "NO PRETRAINING NOW"}
            return None

        train = self.train_data
        valid = self.valid_data
        test = self.test_data
        assert (valid is not None) == ("valid" in schedule["eval"]), (
            "Confusion about validation set!")
        assert (test is not None) == ("test" in schedule["eval"]), (
            "Confusion about test set!")

        # start the pretrain from a certain layer
        # be careful! default starting from layer index 0!
        if schedule["reload_pretrain"]:
            epoch_checker(schedule, schedule['pretrain_before'] - 1)
            # reload the log file
            fname = reload_log(self.logging, schedule, 'layer')
            pp = {"msg": "RELOAD LOG from {}".format(fname)}
            munk.taggify(self.logging, "pretty").send(pp)

        if "pretrain_before" in schedule:
            assert (bool(schedule["pretrain_before"]) == bool(schedule["reload_pretrain"])), (
                "Confusion about pretrain_before and reload! No RELOAD and NOT pretrain from First layer")
            if schedule["pretrain_before"] > len(self.stack):
                pp = {"msg": "pretrain_before OUT of the stack!"}
                munk.taggify(self.logging, "pretty").send(pp)
                return None
        else:
            schedule.update({"pretrain_before": 0})

        for i, (layer, sched) in enumerate(izip(self, self.stack)):
            pt_params = layer.pt_init(**sched)
            pp = {"layer": i, "type": str(layer)}
            munk.taggify(self.logging, "pretty").send(pp)
            log = munk.add_keyvalue(self.logging, "layer", i)
            # performance = munk.add_keyvalue(self.performance, "layer", i)
            opt_schedule = sched["opt"]
            epochs = opt_schedule["epochs"]

            if i < schedule["pretrain_before"]:
                pt_params, reload_epochs = self.reload_one_layer(schedule, i)
                epochs = epochs - reload_epochs
                sched["opt"]["epochs"] = epochs
                if epochs > 0:
                    true_epochs = int(epochs / sched["opt"]["stop"])
                    pp = {"msg": "{} more PRETRAINING EPOCHS of layer {}".format(true_epochs, i)}
                    munk.taggify(self.logging, "pretty").send(pp)

            if epochs > 0:
                opt_schedule["f"] = layer.pt_score
                opt_schedule["fprime"] = layer.pt_grad

                opt, evals, peeks = prepare_opt(opt_schedule, pt_params, schedule, train, valid, test)

                stop = opt_schedule["stop"]
                for j, info in enumerate(opt):
                    if (j+1) % stop == 0:
                        for e in evals:
                            info[e] = evals[e](pt_params)
                        info = replace_gnumpy_data(info)
                        log.send(info)

                    if (j+1) == epochs:
                        break
            else:
                pp = {"msg": "NO PRETRAINING of layer %i NOW"%i}
                munk.taggify(self.logging, "pretty").send(pp)

            info = layer.pt_done(pt_params, **sched)
            pt_params = None
            log.send(info)

            # move data forward, save in temporary hdf5
            if i < (len(self) - 1):
                nxt_name = strftime("%Y-%m-%d-%H:%M:%S") + "_L" + str(i+1) + "_TMP.h5"
                nxt = h5py.File(nxt_name)
                pp = {"msg": "Take care of temporary " + nxt_name}
                munk.taggify(self.logging, "pretty").send(pp)
                # if a validation set is available, move it forward, too.
                if valid:
                    valid[0] = self.next_hdf5(layer, valid[0], "validation", nxt, chunk=512)
                # if a test set is available, move it forward, too.
                if test:
                    test[0] = self.next_hdf5(layer, test[0], "test", nxt, chunk=512)
                train[0] = self.next_hdf5(layer, train[0], "train", nxt, chunk=512)

    def train(self, schedule):

        # make sure reload the pretrain parameters and reload the train parameters won't happen together
        assert (not (schedule["reload_pretrain"] & schedule["reload_train"])), \
            "reload pretrain and reload_train mismatch"

        self.train_data = [schedule["train"][0], schedule["train"][1]]
        train = self.train_data
        valid = (None if not schedule.get("valid")
                 else [schedule["valid"][0], schedule["valid"][1]])
        test = (None if not schedule.get("test")
                else [schedule["test"][0], schedule["test"][1]])
        assert (valid is not None) == ("valid" in schedule["eval"]), "Confusion about validation set!"
        assert (test is not None) == ("test" in schedule["eval"]), "Confusion about test set!"
        opt_schedule = schedule["opt"]

        pp = {"type" : str(self)}

        munk.taggify(self.logging, "pretty").send(pp)


        epochs = opt_schedule["epochs"]

        if schedule["reload_train"]:
            # reload the log file
            fname = reload_log(self.logging, schedule, 'stack')
            pp = {"msg": "RELOAD LOG from {}".format(fname)}
            munk.taggify(self.logging, "pretty").send(pp)

            # reload params
            self.params, reload_epochs = self.reload_stack(schedule)
            for layer, (c1, c2) in izip(self, izip(self.cuts[:-1], self.cuts[1:])):
                layer.p = self.params[c1:c2]
            epochs = epochs - reload_epochs
            schedule["opt"]["epochs"] = epochs
            if epochs > 0:
                true_epochs = int(epochs / schedule["opt"]["stop"])
                pp = {"msg": "{} more TRAINING EPOCHS of the stack".format(true_epochs)}
                munk.taggify(self.logging, "pretty").send(pp)

        log = munk.add_keyvalue(self.logging, "layer", "Stack")

        if epochs > 0:
            opt_schedule["f"] = self.score
            opt_schedule["fprime"] = self.grad

            if "eval_score" in opt_schedule:
                self._eval_score = opt_schedule["eval_score"]
                opt_schedule["eval_score"] = self.evaluate_score

            opt, evals, peeks = prepare_opt(opt_schedule, self.params, schedule, train, valid, test)

            stop = opt_schedule["stop"]
            if "peeks" in opt_schedule:
                peek_iv = opt_schedule["peek_interval"]
                peek_files = {}
                for p in opt_schedule["peeks"]:
                    peek_files[p] = p + ".peek"
            else:
                peek_iv = epochs + 1

            for i, info in enumerate(opt):
                if (i+1) % stop == 0:
                    for e in evals:
                        info[e] = evals[e](self.params)
                    info = replace_gnumpy_data(info)
                    log.send(info)

                if i+1 == epochs:
                    break

                if (i+1) % peek_iv == 0:
                    for p in peeks:
                        prediction, inputs = peeks[p](self.params)
                        np.savez(peek_files[p], prediction, inputs)
                        pp = {"msg": "Writing peek file %s"%peek_files[p]}
                        munk.taggify(self.logging, "pretty").send(pp)

        else:
            pp = {"msg": "NO FINETUNING of stack Now"}
            munk.taggify(self.logging, "pretty").send(pp)

        _params = self.params.as_numpy_array().tolist()
        info = dict(params=_params, shape=self.__repr__())
        log.send(info)

    def score(self, params, inputs, targets, **kwargs):
        data = inputs
        for layer, (c1, c2) in izip(self, izip(self.cuts[:-1], self.cuts[1:])):
            data = layer.fward(self.params[c1:c2], data)
        return self._score(data, targets, **kwargs)

    def grad(self, params, inputs, targets, **kwargs):
        data = inputs
        for layer, (c1, c2) in izip(self, izip(self.cuts[:-1], self.cuts[1:])):
            data = layer.fprop(self.params[c1:c2], data)

        _, delta = self._score(data, targets, error=True)

        g = gzeros(self.psize)
        for layer, (c1, c2) in izip(self[::-1], izip(self.cuts[-2::-1], self.cuts[:0:-1])):
            delta = layer.bprop(params=params[c1:c2], grad=g[c1:c2], delta=delta)
        return g

    def evaluate_score(self, params, inputs, targets, **kwargs):
        data = inputs
        for layer, (c1, c2) in izip(self, izip(self.cuts[:-1], self.cuts[1:])):
            data = layer.fward(self.params[c1:c2], data)
        return self._eval_score(data, targets, **kwargs)

    def next_hdf5(self, layer, data, dname, nxt, chunk):
        """After pretraining one layer, move
        data to new temporary hdf5 store.
        """
        n = data.shape[0]
        d = layer.shape[1]
        tmp = nxt.create_dataset(name=dname, shape=(n, d), dtype=data.dtype)
        for i in xrange(0, n, chunk):
            tmp[i:i+chunk] = layer._fward(data[i:i+chunk])
        return tmp

    def _fward(self, data):
        for layer in self:
            data = layer._fward(data)
        return loss_table[self._score](data, targets=None, predict=True)

    def _fward_layers(self, data, layers):
        """
        Only pass _data_ through _layers_ many layers.
        No loss applied!.
        """
        for layer in self[:layers]:
            data = layer._fward(data)
        return data

    # def reload_stack(self, depot, folder, tag, schedule):
    #     """
    #     reload schedule and parameters from depot/folder/tag.params.
    #     depot, abs path.
    #     """
    #     from utils import load_params
    #     from os.path import join
    #     from gnumpy import as_garray
    #     fname = join(depot, folder, tag + ".params")
    #     params = load_params(fname)
    #     params_stack = params['Stack']['params']
    #     self.params = as_garray(params_stack)
    #     for (layer, sched), (c1, c2) in izip(
    #             izip(self, self.stack), izip(self.cuts[:-1], self.cuts[1:])):
    #         layer.pt_init(**sched)
    #         layer.p = self.params[c1:c2]
    #     pp = {"msg": "RELOAD PARAMS from {}".format(fname)}
    #     munk.taggify(self.logging, "pretty").send(pp)

    def reload_stack(self, schedule):
        """
        reload schedule and parameters from depot/folder/tag.params.
        depot, abs path.
        """
        from gnumpy import as_garray

        fname, params, reload_epochs = reload_checker(schedule, 'stack')
        stack_params = params['Stack']['params']
        pp = {"msg": "RELOAD STACK PARAMS from {}".format(fname)}
        munk.taggify(self.logging, "pretty").send(pp)

        return as_garray(stack_params), reload_epochs

    def reload_layers(self, schedule):
        """
        reload schedule and parameters from depot/folder/tag.params.
        depot, abs path.
        only load 'pretraining' parameters through _layers_ many layers.
        """
        from gnumpy import as_garray
        epoch_checker(schedule, schedule['pretrain_before'] - 1)
        fname, params = self.reload_checker(schedule,
                                            tuple(range(schedule['pretrain_before'])))

        train = self.train_data
        valid = self.valid_data
        test = self.test_data

        assert (valid is not None) == ("valid" in schedule["eval"]),\
            "Confusion about validation set!"
        assert (test is not None) == ("test" in schedule["eval"]),\
            "Confusion about test set!"

        for i, (layer, sched) in enumerate(
                izip(self[:schedule["pretrain_before"]],
                     self.stack[:schedule["pretrain_before"]])):
            l = len(layer.p)
            layer.pt_init(**sched)
            pt_params = as_garray(params[i]['params'])
            layer.p = pt_params[:l]

            pp = {"layer": i, "type": str(layer)}
            munk.taggify(self.logging, "pretty").send(pp)
            log = munk.add_keyvalue(self.logging, "layer", i)
            epochs = schedule["opt"]["epochs"]
            if epochs > 0:
                pp = {"msg": "RELOAD PARAMS from {}".format(fname)}
                munk.taggify(self.logging, "pretty").send(pp)
                pass
            else:
                pp = {"msg": "NO PRETRAINING of layer %i"%i}
                munk.taggify(self.logging, "pretty").send(pp)
            info = layer.pt_done(pt_params, **sched)
            pt_params = None
            log.send(info)

            # move data forward, save in temporary hdf5
            if i < (len(self) - 1):
                nxt_name = (strftime("%Y-%m-%d-%H:%M:%S") + "_L"
                            + str(i+1) + "_TMP.h5")
                nxt = h5py.File(nxt_name)
                pp = {"msg": "Take care of temporary " + nxt_name}
                munk.taggify(self.logging, "pretty").send(pp)
                # if a validation set is available, move it forward, too.
                if valid:
                    self.valid_data[0] = (self.next_hdf5(
                        layer, self.valid_data[0], "validation",
                        nxt, chunk=512))
                # if a test set is available, move it forward, too.
                if test:
                    self.test_data[0] = (self.next_hdf5(
                        layer, self.test_data[0], "test",
                        nxt, chunk=512))
                self.train_data[0] = (self.next_hdf5(
                    layer, self.train_data[0], "train", nxt, chunk=512))

    def reload_one_layer(self, schedule, l):
        '''
        reload schedule and parameters from depot/folder/tag.params.
        depot, abs path.
        reload only l-th layer
        return file name, reloaded params and reloaded epochs
        '''
        from gnumpy import as_garray

        fname, params, reload_epochs = reload_checker(schedule, 'layer', l)
        # (layer, sched) = izip(self[l], self.stack[l])
        # length = len(self[l].p)
        pt_params = params[l]['params']
        pp = {"msg": "RELOAD LAYER {} PARAMS from {}".format(l, fname)}
        munk.taggify(self.logging, "pretty").send(pp)

        # return fname, as_garray(pt_params[:length]), reload_epochs
        return as_garray(pt_params), reload_epochs
