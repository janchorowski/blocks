from abc import abstractmethod, ABCMeta


class LogMonitor(object):
    """A LogMonitor stores all monitor channels computed over training.
    """
    __meta__ = ABCMeta

    @abstractmethod
    def append(self, num_examples_seen, context, name, value):
        """Append one record to the monitor log
        """

    def append_many(self, num_examples_seen, context, name_value_pairs):
        for name, value in name_value_pairs:
            self.append(num_examples_seen, context, name, value)

    def append_dict(self, num_examples_seen, context, name_value_dict):
        return self.append_many(num_examples_seen, context,
                                name_value_dict.iteritems())


class CSVLogMonitor(LogMonitor):
    def __init__(self, fname, append=True, **kwargs):
        super(CSVLogMonitor, self).__init__(**kwargs)
        self.file = None
        if append:
            mode = 'at'
        else:
            mode = 'wt'
        self.file = open(fname, mode)

        # detect if file is empty, and if yes write the header
        self.file.seek(0, 2)
        if self.file.tell() == 0L:
            self.file.write('num_examples_seen,context,name,value\n')

        # used to trigger a flush whenever we start getting data
        # for the next batch
        self._num_examples_seen = None

    def append(self, num_examples_seen, context, name, value):
        if self._num_examples_seen != num_examples_seen:
            self._num_examples_seen = num_examples_seen
            self.file.flush()
        self.write('%(num_examples_seen)s,%(context)s,%(name)s,%(value)\n' % locals())

    def append_many(self, num_examples_seen, context, name_value_pairs):
        if self._num_examples_seen != num_examples_seen:
            self._num_examples_seen = num_examples_seen
            self.file.flush()
        for name, value in name_value_pairs:
            self.write('%(num_examples_seen)s,%(context)s,%(name)s,%(value)\n' % locals())

    def close(self):
        if self.file:
            self.file.close()
            self.file = None


class ConsoleLogMonitor(LogMonitor):
    def __init__(self, **kwargs):
        super(ConsoleLogMonitor, self).__init__(**kwargs)
        # used to trigger a flush whenever we start getting data for the
        # next batch
        self._num_examples_seen = None
        self._context = None
        self._name_values = []

    def append(self, num_examples_seen, context, name, value):
        if (self._num_examples_seen != num_examples_seen
            or self._context != context):
            self._num_examples_seen = num_examples_seen
            self._context = context
            message = ['%(num_examples_seen)s,%(context)s:' % locals()]
            for name_val in self._name_values:
                message.append('%s=%.3g' % name_val)
            print ' '.join(message)
            self._name_values = [(name, value)]
        else:
            self._name_values.append((name, value))
