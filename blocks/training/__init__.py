"""Training loops.

"""
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from copy import copy

from blocks.select import BrickSelection
from blocks.bricks import lazy
from blocks.utils import update_instance
from blocks.training.aggregators import (DatasetEvaluator,
                                         MinibatchEvaluator)


class Model(object):
    """A pickleable model that can be trained

    """
    __meta__ = ABCMeta

    @abstractmethod
    def get_cost(self):
        """Return the Theano variable indicating the full training cost

        """
        pass

    def censor_updates(self, updates):
        """Modify training updates to enforce constraints.

        """
        return updates


class TrainState(object):
    """The state of the training.

    It will be passed to each training extension. Extensions are free to
    modify or add to it.

    This is the main mechanism for passing information between extensions.

    The `expressions` dict will be filled by the :class:`ComputeExpressions`
    extension, which is always run first. Its keys will be context names,
    during :meth:`MainLoopExtension.on_mini_batch` this will be typically
    'in_training', while for other callbacks these will be the dataset
    names passed to the MainLoop and in turn to the
    :class:`ComputeExpressions` extension. Each entry in the dictionary
    will again be a dict from theano expressions to their values.

    The `additional_monitors` is a list of (name,value) pairs that will
    be appended to the monitor by the monitoring extension, which is always
    run last.

    """
    IN_TRAINING = 'in_training'

    def __init__(self, **kwargs):
        super(TrainState, self).__init__(**kwargs)
        self.number_batches_seen = 0
        self.number_epochs_seen = 0
        self.number_batches_seen_in_epoch = 0
        self.expressions = defaultdict(lambda: {})
        self.additional_monitors = []
        self.continue_training = True


class MainLoopExtension(object):
    """Provide functions to be called at valid moments during training.

    The extensions will be called sorted by their priority (lower is first).

    Before the extension is run, theano variables given by the
    :meth:`get_required_expressions` will be evaluated, as described for
    the class :class:`TrainState`.

    Parameters
    ----------
    priority : int
        the priority with which the extensions callbacks are run.
        Lower priorities are evaluated sooner.

    """
    def __init__(self, priority=0):
        self.priority = priority

    def on_traininig_start(self, state):
        """Called when training starts.

        """
        pass

    def on_training_end(self, state):
        """Called when training ends.

        """
        pass

    def on_epoch(self, state):
        """Called after each epoch.

        """
        pass

    def on_timer(self, state):
        """Called periodically in between epochs.

        """
        pass

    def on_minibatch(self, state):
        """Called by stochastic trainers after each minibatch.

        """
        pass

    def get_required_expressions(self):
        """List expressions to be computed before callbacks are run.

        """
        return []

    def get_minibatch_updates(self):
        """List updates to be run during fprop through the model.

        """
        return []

    def censor_minibatch_updates(self, updates):
        """Modify updates to model parameters.

        """
        return updates


class StateInitializer(MainLoopExtension):
    def __init__(self):
        MainLoopExtension.__init__(self, priority=None)
        self.number_batches_seen = 0
        self.number_epochs_seen = 0
        self.number_batches_seen_in_epoch = 0

    def initialize_state(self, state):
        state.number_batches_seen = self.number_batches_seen
        state.number_epochs_seen = self.number_epochs_seen
        state.number_batches_seen_in_epoch = self.number_batches_seen_in_epoch
        state.expressions = defaultdict(lambda: {})
        state.additional_monitors = []
        state.continue_training = True

    def on_epoch(self, state):
        self.number_epochs_seen += 1
        self.number_batches_seen_in_epoch = 0

    def on_minibatch(self, state):
        self.number_batches_seen += 1
        self.number_batches_seen_in_epoch += 1

    def get_minibatch_updates(self):
        return MainLoopExtension.get_minibatch_updates(self)


class ComputeExpressions(MainLoopExtension):
    def __init__(self, expressions, datasets, **kwargs):
        # in Python None is smaller than any int or float.
        super(ComputeExpressions, self).__init__(priority=None, **kwargs)
        self.expressions = expressions
        self.datasets = datasets
        self.dataset_evaluator = DatasetEvaluator(self.expressions)
        self.minibatch_evaluator = MinibatchEvaluator(self.expressions)

    def get_required_expressions(self):
        return []

    def get_minibatch_updates(self):
        return self.minibatch_evaluator.updates()

    def compute_all(self, state):
        for name, view in self.datasets.iteritems():
            expressions = self.dataset_evaluator.evaluate(view)
            state.expressions[name] = expressions

    def on_traininig_start(self, state):
        return self.monitor_all(state)

    def on_training_end(self, state):
        return self.monitor_all(state)

    def on_epoch(self, state):
        return self.monitor_all(state)

    def on_timer(self, state):
        return self.monitor_all(state)

    def on_minibatch(self, state):
        expressions = self.minibatch_evaluator.read_expressions()
        state.expressions[TrainState.IN_TRAINING] = expressions


class StochasticTrainLoop(object):
    """Runs a loop for one epoch of minibatch training.

    This class should be sub-classed by all stochastic trainers.
    """
    def __init__(self, extensions, **kwargs):
        super(StochasticTrainLoop, self).__init__(**kwargs)
        self.extensions = extensions

    def do_one_iteration(self, batch):
        raise NotImplementedError()

    def __call__(self, dataset, state):
        for batch in dataset:
            self.do_one_iteration(batch)
            state.add_batch()
            for e in self.extensions:
                e.on_minibatch(state)


class TrainLoop(object):
    def __init__(self,
                 model,
                 trainer,
                 extensions
                 ):
        update_instance(self, locals())
        self.train_state = TrainState()

    def train(self, train_set_view):
        extensions = sorted(self.extensions, key=lambda x: x.priority)
        epoch_train_fun = self.trainer.get_epoch_train_fun()
        state = self.state
        assert isinstance(extensions[0], ComputeExpressions)

        for e in extensions:
            e.on_traininig_start(state)
        state.clean()  # clean the state, keeping only its initial fields

        try:
            while state.continue_training:
                epoch_train_fun(train_set_view, state, extensions)
                state.add_epoch()
                for e in extensions:
                    e.on_epoch(state)
        finally:
            state.clean()
            for e in extensions:
                e.on_traininig_end(state)
