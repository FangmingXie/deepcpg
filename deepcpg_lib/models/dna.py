"""DNA models.

Provides models trained with DNA sequence windows.
"""

from __future__ import division
from __future__ import print_function

import inspect

from keras import layers as kl
from keras import regularizers as kr

from .utils import Model
from ..utils import get_from_module


class DnaModel(Model):
    """Abstract class of a DNA model."""

    def __init__(self, *args, **kwargs):
        super(DnaModel, self).__init__(*args, **kwargs)
        self.scope = 'dna'

    def inputs(self, dna_wlen):
        return [kl.Input(shape=(dna_wlen, 4), name='dna')]

# added by Fangming 08-04-2017
class CnnL1f100L1h350(DnaModel):
    """CNN with one convolutional and one fully-connected layer with 128 units.

    .. code::

        Parameters: ? 
        Specification: conv[100@11]_mp[4]_fc[350]_do
    """

    def __init__(self, nb_hidden=350, *args, **kwargs):
        super(CnnL1f100L1h350, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(100, 11, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        x = kl.Flatten()(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)

# added by Fangming 07-31-2017
class CnnL1f200L1h350(DnaModel):
    """CNN with one convolutional and one fully-connected layer with 128 units.

    .. code::

        Parameters: ? 
        Specification: conv[200@11]_mp[4]_fc[350]_do
    """

    def __init__(self, nb_hidden=350, *args, **kwargs):
        super(CnnL1f200L1h350, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(200, 11, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        x = kl.Flatten()(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)

# added by Fangming 08-04-2017
class CnnL1f300L1h350(DnaModel):
    """CNN with one convolutional and one fully-connected layer with 128 units.

    .. code::

        Parameters: ? 
        Specification: conv[300@11]_mp[4]_fc[350]_do
    """

    def __init__(self, nb_hidden=350, *args, **kwargs):
        super(CnnL1f300L1h350, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(300, 11, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        x = kl.Flatten()(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)

# added by Fangming 08-04-2017
class CnnL1f400L1h350(DnaModel):
    """CNN with one convolutional and one fully-connected layer with 128 units.

    .. code::

        Parameters: ? 
        Specification: conv[400@11]_mp[4]_fc[350]_do
    """

    def __init__(self, nb_hidden=350, *args, **kwargs):
        super(CnnL1f400L1h350, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(400, 11, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        x = kl.Flatten()(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)

# added by Fangming 08-04-2017
class CnnL1f500L1h350(DnaModel):
    """CNN with one convolutional and one fully-connected layer with 128 units.

    .. code::

        Parameters: ? 
        Specification: conv[500@11]_mp[4]_fc[350]_do
    """

    def __init__(self, nb_hidden=350, *args, **kwargs):
        super(CnnL1f500L1h350, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(500, 11, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        x = kl.Flatten()(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)

# added by Fangming 08-04-2017
class CnnL1f600L1h350(DnaModel):
    """CNN with one convolutional and one fully-connected layer with 128 units.

    .. code::

        Parameters: ? 
        Specification: conv[600@11]_mp[4]_fc[350]_do
    """

    def __init__(self, nb_hidden=350, *args, **kwargs):
        super(CnnL1f600L1h350, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(600, 11, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        x = kl.Flatten()(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)


# added by Fangming 08-04-2017
class CnnL1f700L1h350(DnaModel):
    """CNN with one convolutional and one fully-connected layer with 128 units.

    .. code::

        Parameters: ? 
        Specification: conv[700@11]_mp[4]_fc[350]_do
    """

    def __init__(self, nb_hidden=350, *args, **kwargs):
        super(CnnL1f700L1h350, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(700, 11, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        x = kl.Flatten()(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)

# added by Fangming 08-04-2017
class CnnL1f800L1h350(DnaModel):
    """CNN with one convolutional and one fully-connected layer with 128 units.

    .. code::

        Parameters: ? 
        Specification: conv[800@11]_mp[4]_fc[350]_do
    """

    def __init__(self, nb_hidden=350, *args, **kwargs):
        super(CnnL1f800L1h350, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(800, 11, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        x = kl.Flatten()(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)

# added by Fangming 08-04-2017
class CnnL1f900L1h350(DnaModel):
    """CNN with one convolutional and one fully-connected layer with 128 units.

    .. code::

        Parameters: ? 
        Specification: conv[900@11]_mp[4]_fc[350]_do
    """

    def __init__(self, nb_hidden=350, *args, **kwargs):
        super(CnnL1f900L1h350, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(900, 11, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        x = kl.Flatten()(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)

# added by Fangming 08-04-2017
class CnnL1f100L2h350(DnaModel):
    """CNN with one convolutional and one fully-connected layer with 128 units.

    .. code::

        Parameters: ? 
        Specification: conv[100@11]_mp[4]_fc[350]_fc[350]_do
    """

    def __init__(self, nb_hidden=350, *args, **kwargs):
        super(CnnL1f100L2h350, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(100, 11, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        x = kl.Flatten()(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)

# added by Fangming 07-31-2017
class CnnL1f200L2h350(DnaModel):
    """CNN with one convolutional and one fully-connected layer with 128 units.

    .. code::

        Parameters: ? 
        Specification: conv[200@11]_mp[4]_fc[350]_fc[350]_do
    """

    def __init__(self, nb_hidden=350, *args, **kwargs):
        super(CnnL1f200L2h350, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(200, 11, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        x = kl.Flatten()(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)

# added by Fangming 08-04-2017
class CnnL1f300L2h350(DnaModel):
    """CNN with one convolutional and one fully-connected layer with 128 units.

    .. code::

        Parameters: ? 
        Specification: conv[300@11]_mp[4]_fc[350]_fc[350]_do
    """

    def __init__(self, nb_hidden=350, *args, **kwargs):
        super(CnnL1f300L2h350, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(300, 11, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        x = kl.Flatten()(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)

# added by Fangming 08-04-2017
class CnnL1f400L2h350(DnaModel):
    """CNN with one convolutional and one fully-connected layer with 128 units.

    .. code::

        Parameters: ? 
        Specification: conv[400@11]_mp[4]_fc[350]_fc[350]_do
    """

    def __init__(self, nb_hidden=350, *args, **kwargs):
        super(CnnL1f400L2h350, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(400, 11, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        x = kl.Flatten()(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)

# added by Fangming 08-04-2017
class CnnL1f500L2h350(DnaModel):
    """CNN with one convolutional and one fully-connected layer with 128 units.

    .. code::

        Parameters: ? 
        Specification: conv[500@11]_mp[4]_fc[350]_fc[350]_do
    """

    def __init__(self, nb_hidden=350, *args, **kwargs):
        super(CnnL1f500L2h350, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(500, 11, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        x = kl.Flatten()(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)

# added by Fangming 08-04-2017
class CnnL1f600L2h350(DnaModel):
    """CNN with one convolutional and one fully-connected layer with 128 units.

    .. code::

        Parameters: ? 
        Specification: conv[600@11]_mp[4]_fc[350]_fc[350]_do
    """

    def __init__(self, nb_hidden=350, *args, **kwargs):
        super(CnnL1f600L2h350, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(600, 11, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        x = kl.Flatten()(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)


# added by Fangming 08-04-2017
class CnnL1f700L2h350(DnaModel):
    """CNN with one convolutional and one fully-connected layer with 128 units.

    .. code::

        Parameters: ? 
        Specification: conv[700@11]_mp[4]_fc[350]_fc[350]_do
    """

    def __init__(self, nb_hidden=350, *args, **kwargs):
        super(CnnL1f700L2h350, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(700, 11, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        x = kl.Flatten()(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)

# added by Fangming 08-04-2017
class CnnL1f800L2h350(DnaModel):
    """CNN with one convolutional and one fully-connected layer with 128 units.

    .. code::

        Parameters: ? 
        Specification: conv[800@11]_mp[4]_fc[350]_fc[350]_do
    """

    def __init__(self, nb_hidden=350, *args, **kwargs):
        super(CnnL1f800L2h350, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(800, 11, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        x = kl.Flatten()(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)

# added by Fangming 08-04-2017
class CnnL1f900L2h350(DnaModel):
    """CNN with one convolutional and one fully-connected layer with 128 units.

    .. code::

        Parameters: ? 
        Specification: conv[900@11]_mp[4]_fc[350]_fc[350]_do
    """

    def __init__(self, nb_hidden=350, *args, **kwargs):
        super(CnnL1f900L2h350, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(900, 11, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        x = kl.Flatten()(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)
        
# added by Fangming 08-04-2017
class CnnL1f100L2h128(DnaModel):
    """CNN with one convolutional and one fully-connected layer with 128 units.

    .. code::

        Parameters: ? 
        Specification: conv[100@11]_mp[4]_fc[128]_fc[128]_do
    """

    def __init__(self, nb_hidden=128, *args, **kwargs):
        super(CnnL1f100L2h128, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(100, 11, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        x = kl.Flatten()(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)

# added by Fangming 07-31-2017
class CnnL1f200L2h128(DnaModel):
    """CNN with one convolutional and one fully-connected layer with 128 units.

    .. code::

        Parameters: ? 
        Specification: conv[200@11]_mp[4]_fc[128]_fc[128]_do
    """

    def __init__(self, nb_hidden=128, *args, **kwargs):
        super(CnnL1f200L2h128, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(200, 11, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        x = kl.Flatten()(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)

# added by Fangming 08-04-2017
class CnnL1f300L2h128(DnaModel):
    """CNN with one convolutional and one fully-connected layer with 128 units.

    .. code::

        Parameters: ? 
        Specification: conv[300@11]_mp[4]_fc[128]_fc[128]_do
    """

    def __init__(self, nb_hidden=128, *args, **kwargs):
        super(CnnL1f300L2h128, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(300, 11, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        x = kl.Flatten()(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)

# added by Fangming 08-04-2017
class CnnL1f400L2h128(DnaModel):
    """CNN with one convolutional and one fully-connected layer with 128 units.

    .. code::

        Parameters: ? 
        Specification: conv[400@11]_mp[4]_fc[128]_fc[128]_do
    """

    def __init__(self, nb_hidden=128, *args, **kwargs):
        super(CnnL1f400L2h128, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(400, 11, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        x = kl.Flatten()(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)

# added by Fangming 08-04-2017
class CnnL1f500L2h128(DnaModel):
    """CNN with one convolutional and one fully-connected layer with 128 units.

    .. code::

        Parameters: ? 
        Specification: conv[500@11]_mp[4]_fc[128]_fc[128]_do
    """

    def __init__(self, nb_hidden=128, *args, **kwargs):
        super(CnnL1f500L2h128, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(500, 11, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        x = kl.Flatten()(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)

# added by Fangming 08-04-2017
class CnnL1f600L2h128(DnaModel):
    """CNN with one convolutional and one fully-connected layer with 128 units.

    .. code::

        Parameters: ? 
        Specification: conv[600@11]_mp[4]_fc[128]_fc[128]_do
    """

    def __init__(self, nb_hidden=128, *args, **kwargs):
        super(CnnL1f600L2h128, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(600, 11, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        x = kl.Flatten()(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)


# added by Fangming 08-04-2017
class CnnL1f700L2h128(DnaModel):
    """CNN with one convolutional and one fully-connected layer with 128 units.

    .. code::

        Parameters: ? 
        Specification: conv[700@11]_mp[4]_fc[128]_fc[128]_do
    """

    def __init__(self, nb_hidden=128, *args, **kwargs):
        super(CnnL1f700L2h128, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(700, 11, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        x = kl.Flatten()(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)

# added by Fangming 08-04-2017
class CnnL1f800L2h128(DnaModel):
    """CNN with one convolutional and one fully-connected layer with 128 units.

    .. code::

        Parameters: ? 
        Specification: conv[800@11]_mp[4]_fc[128]_fc[128]_do
    """

    def __init__(self, nb_hidden=128, *args, **kwargs):
        super(CnnL1f800L2h128, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(800, 11, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        x = kl.Flatten()(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)

# added by Fangming 08-04-2017
class CnnL1f900L2h128(DnaModel):
    """CNN with one convolutional and one fully-connected layer with 128 units.

    .. code::

        Parameters: ? 
        Specification: conv[900@11]_mp[4]_fc[128]_fc[128]_do
    """

    def __init__(self, nb_hidden=128, *args, **kwargs):
        super(CnnL1f900L2h128, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(900, 11, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        x = kl.Flatten()(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)


# added by Fangming 07-29-2017
class CnnL1f128L2h128(DnaModel):
    """CNN with one convolutional and one fully-connected layer with 128 units.

    .. code::

        Parameters: ? 
        Specification: conv[128@11]_mp[4]_fc[128]_fc[128]_do
    """

    def __init__(self, nb_hidden=128, *args, **kwargs):
        super(CnnL1f128L2h128, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(128, 11, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        x = kl.Flatten()(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)


# added by Fangming 07-29-2017
class CnnL1f256L2h128(DnaModel):
    """CNN with one convolutional and one fully-connected layer with 128 units.

    .. code::

        Parameters: ? 
        Specification: conv[256@11]_mp[4]_fc[128]_fc[128]_do
    """

    def __init__(self, nb_hidden=128, *args, **kwargs):
        super(CnnL1f256L2h128, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(256, 11, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        x = kl.Flatten()(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)

# added by Fangming 07-30-2017
class CnnL1f512L2h128(DnaModel):
    """CNN with one convolutional and one fully-connected layer with 128 units.

    .. code::

        Parameters: ? 
        Specification: conv[512@11]_mp[4]_fc[128]_fc[128]_do
    """

    def __init__(self, nb_hidden=128, *args, **kwargs):
        super(CnnL1f512L2h128, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(512, 11, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        x = kl.Flatten()(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)


class CnnL1h128(DnaModel):
    """CNN with one convolutional and one fully-connected layer with 128 units.

    .. code::

        Parameters: 4,100,000
        Specification: conv[128@11]_mp[4]_fc[128]_do
    """

    def __init__(self, nb_hidden=128, *args, **kwargs):
        super(CnnL1h128, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(128, 11, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        x = kl.Flatten()(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)


class CnnL1h256(CnnL1h128):
    """CNN with one convolutional and one fully-connected layer with 256 units.

    .. code::

        Parameters: 8,100,000
        Specification: conv[128@11]_mp[4]_fc[256]_do
    """

    def __init__(self,  *args, **kwargs):
        super(CnnL1h256, self).__init__(*args, **kwargs)
        self.nb_hidden = 256


# added by Fangming 07-25-2017
class CnnL1h320(DnaModel):
    """CNN with one convolutional and one fully-connected layer with 128 units.

    .. code::

        Parameters: ? 
        Specification: conv[320@11]_mp[4]_fc[320]_do
    """

    def __init__(self, nb_hidden=320, *args, **kwargs):
        super(CnnL1h320, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(320, 11, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        x = kl.Flatten()(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)


# added by Fangming 07-25-2017
class CnnL1h512(DnaModel):
    """CNN with one convolutional and one fully-connected layer with 128 units.

    .. code::

        Parameters: ? 
        Specification: conv[512@11]_mp[4]_fc[512]_do
    """

    def __init__(self, nb_hidden=512, *args, **kwargs):
        super(CnnL1h512, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(512, 11, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        x = kl.Flatten()(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)


class CnnL2h128(DnaModel):
    """CNN with two convolutional and one fully-connected layer with 128 units.

    .. code::

        Parameters: 4,100,000
        Specification: conv[128@11]_mp[4]_conv[256@3]_mp[2]_fc[128]_do
    """

    def __init__(self, nb_hidden=128, *args, **kwargs):
        super(CnnL2h128, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(128, 11, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(256, 3, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(2)(x)

        x = kl.Flatten()(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)


class CnnL2h256(CnnL2h128):
    """CNN with two convolutional and one fully-connected layer with 256 units.

    .. code::

        Parameters: 8,100,000
        Specification: conv[128@11]_mp[4]_conv[256@3]_mp[2]_fc[256]_do
    """

    def __init__(self,  *args, **kwargs):
        super(CnnL2h256, self).__init__(*args, **kwargs)
        self.nb_hidden = 256

# added by fangming 7/24/2017
class CnnL2h320(DnaModel):
    """CNN with two convolutional and one fully-connected layer with 128 units.

    .. code::

        Parameters: ?? 
        Specification: conv[320@11]_mp[4]_conv[640@3]_mp[2]_fc[320]_do
    """

    def __init__(self, nb_hidden=320, *args, **kwargs):
        super(CnnL2h320, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(320, 11, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(640, 3, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(2)(x)

        x = kl.Flatten()(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)

# added by fangming 7/24/2017
class CnnL2h512(DnaModel):
    """CNN with two convolutional and one fully-connected layer with 128 units.

    .. code::

        Parameters: ?? 
        Specification: conv[320@11]_mp[4]_conv[640@3]_mp[2]_fc[320]_do
    """

    def __init__(self, nb_hidden=512, *args, **kwargs):
        super(CnnL2h512, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(512, 11, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(1024, 3, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(2)(x)

        x = kl.Flatten()(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)



class CnnL3h128(DnaModel):
    """CNN with three convolutional and one fully-connected layer with 128 units.

    .. code::

        Parameters: 4,400,000
        Specification: conv[128@11]_mp[4]_conv[256@3]_mp[2]_conv[512@3]_mp[2]_
                       fc[128]_do
    """

    def __init__(self, nb_hidden=128, *args, **kwargs):
        super(CnnL3h128, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(128, 11, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(256, 3, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(2)(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(512, 3, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(2)(x)

        x = kl.Flatten()(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)


class CnnL3h256(CnnL3h128):
    """CNN with three convolutional and one fully-connected layer with 256 units.

    .. code::

        Parameters: 8,300,000
        Specification: conv[128@11]_mp[4]_conv[256@3]_mp[2]_conv[512@3]_mp[2]_
                       fc[256]_do
    """

    def __init__(self,  *args, **kwargs):
        super(CnnL3h256, self).__init__(*args, **kwargs)
        self.nb_hidden = 256

# added by Fangming 7-26-2017
class CnnL3h320(DnaModel):
    """CNN with three convolutional and one fully-connected layer with 128 units.

    .. code::

        Parameters: ? 
        Specification: conv[320@11]_mp[4]_conv[640@3]_mp[2]_conv[1280@3]_mp[2]_
                       fc[128]_do
    """

    def __init__(self, nb_hidden=128, *args, **kwargs):
        super(CnnL3h320, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(320, 11, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(640, 3, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(2)(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(1280, 3, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(2)(x)

        x = kl.Flatten()(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(self.nb_hidden, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)



class CnnRnn01(DnaModel):
    """Convolutional-recurrent model.

    Convolutional-recurrent model with two convolutional layers followed by a
    bidirectional GRU layer.

    .. code::

        Parameters: 1,100,000
        Specification: conv[128@11]_pool[4]_conv[256@7]_pool[4]_bgru[256]_do
    """

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(128, 11, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(256, 7, init=self.init, W_regularizer=w_reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Bidirectional(kl.recurrent.GRU(256, W_regularizer=w_reg))(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)


class ResNet01(DnaModel):
    """Residual network with bottleneck residual units.

    .. code::

        Parameters: 1,700,000
        Specification: conv[128@11]_mp[2]_resb[2x128|2x256|2x512|1x1024]_gap_do

    He et al., 'Identity Mappings in Deep Residual Networks.'
    """

    def _res_unit(self, inputs, nb_filter, size=3, stride=1, stage=1, block=1):

        name = '%02d-%02d/' % (stage, block)
        id_name = '%sid_' % (name)
        res_name = '%sres_' % (name)

        # Residual branch

        # 1x1 down-sample conv
        x = kl.BatchNormalization(name=res_name + 'bn1')(inputs)
        x = kl.Activation('relu', name=res_name + 'act1')(x)
        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(nb_filter[0], 1,
                      name=res_name + 'conv1',
                      subsample_length=stride,
                      init=self.init,
                      W_regularizer=w_reg)(x)

        # LxL conv
        x = kl.BatchNormalization(name=res_name + 'bn2')(x)
        x = kl.Activation('relu', name=res_name + 'act2')(x)
        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(nb_filter[1], size,
                      name=res_name + 'conv2',
                      border_mode='same',
                      init=self.init,
                      W_regularizer=w_reg)(x)

        # 1x1 up-sample conv
        x = kl.BatchNormalization(name=res_name + 'bn3')(x)
        x = kl.Activation('relu', name=res_name + 'act3')(x)
        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(nb_filter[2], 1,
                      name=res_name + 'conv3',
                      init=self.init,
                      W_regularizer=w_reg)(x)

        # Identity branch
        if nb_filter[-1] != inputs._keras_shape[-1] or stride > 1:
            w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
            identity = kl.Conv1D(nb_filter[2], 1,
                                 name=id_name + 'conv1',
                                 subsample_length=stride,
                                 init=self.init,
                                 W_regularizer=w_reg)(inputs)
        else:
            identity = inputs

        x = kl.merge([identity, x], name=name + 'merge', mode='sum')

        return x

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(128, 11,
                      name='conv1',
                      init=self.init,
                      W_regularizer=w_reg)(x)
        x = kl.BatchNormalization(name='bn1')(x)
        x = kl.Activation('relu', name='act1')(x)
        x = kl.MaxPooling1D(2, name='pool1')(x)

        # 124
        x = self._res_unit(x, [32, 32, 128], stage=1, block=1, stride=2)
        x = self._res_unit(x, [32, 32, 128], stage=1, block=2)

        # 64
        x = self._res_unit(x, [64, 64, 256], stage=2, block=1, stride=2)
        x = self._res_unit(x, [64, 64, 256], stage=2, block=2)

        # 32
        x = self._res_unit(x, [128, 128, 512], stage=3, block=1, stride=2)
        x = self._res_unit(x, [128, 128, 512], stage=3, block=2)

        # 16
        x = self._res_unit(x, [256, 256, 1024], stage=4, block=1, stride=2)

        x = kl.GlobalAveragePooling1D()(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)


class ResNet02(ResNet01):
    """Residual network with bottleneck residual units.

    .. code::

        Parameters: 2,000,000
        Specification: conv[128@11]_mp[2]_resb[3x128|3x256|3x512|1x1024]_gap_do

    He et al., 'Identity Mappings in Deep Residual Networks.'
    """

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(128, 11,
                      name='conv1',
                      init=self.init,
                      W_regularizer=w_reg)(x)
        x = kl.BatchNormalization(name='bn1')(x)
        x = kl.Activation('relu', name='act1')(x)
        x = kl.MaxPooling1D(2, name='pool1')(x)

        # 124
        x = self._res_unit(x, [32, 32, 128], stage=1, block=1, stride=2)
        x = self._res_unit(x, [32, 32, 128], stage=1, block=2)
        x = self._res_unit(x, [32, 32, 128], stage=1, block=3)

        # 64
        x = self._res_unit(x, [64, 64, 256], stage=2, block=1, stride=2)
        x = self._res_unit(x, [64, 64, 256], stage=2, block=2)
        x = self._res_unit(x, [64, 64, 256], stage=2, block=3)

        # 32
        x = self._res_unit(x, [128, 128, 512], stage=3, block=1, stride=2)
        x = self._res_unit(x, [128, 128, 512], stage=3, block=2)
        x = self._res_unit(x, [128, 128, 512], stage=3, block=3)

        # 16
        x = self._res_unit(x, [256, 256, 1024], stage=4, block=1, stride=2)

        x = kl.GlobalAveragePooling1D()(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)


class ResConv01(ResNet01):
    """Residual network with two convolutional layers in each residual unit.

    .. code::

        Parameters: 2,800,000
        Specification: conv[128@11]_mp[2]_resc[2x128|1x256|1x256|1x512]_gap_do

    He et al., 'Identity Mappings in Deep Residual Networks.'
    """

    def _res_unit(self, inputs, nb_filter, size=3, stride=1, stage=1, block=1):

        name = '%02d-%02d/' % (stage, block)
        id_name = '%sid_' % (name)
        res_name = '%sres_' % (name)

        # Residual branch
        x = kl.BatchNormalization(name=res_name + 'bn1')(inputs)
        x = kl.Activation('relu', name=res_name + 'act1')(x)
        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(nb_filter, size,
                      name=res_name + 'conv1',
                      border_mode='same',
                      subsample_length=stride,
                      init=self.init,
                      W_regularizer=w_reg)(x)

        x = kl.BatchNormalization(name=res_name + 'bn2')(x)
        x = kl.Activation('relu', name=res_name + 'act2')(x)
        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(nb_filter, size,
                      name=res_name + 'conv2',
                      border_mode='same',
                      init=self.init,
                      W_regularizer=w_reg)(x)

        # Identity branch
        if nb_filter != inputs._keras_shape[-1] or stride > 1:
            w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
            identity = kl.Conv1D(nb_filter, size,
                                 name=id_name + 'conv1',
                                 border_mode='same',
                                 subsample_length=stride,
                                 init=self.init,
                                 W_regularizer=w_reg)(inputs)
        else:
            identity = inputs

        x = kl.merge([identity, x], name=name + 'merge', mode='sum')

        return x

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(128, 11,
                      name='conv1',
                      init=self.init,
                      W_regularizer=w_reg)(x)
        x = kl.BatchNormalization(name='bn1')(x)
        x = kl.Activation('relu', name='act1')(x)
        x = kl.MaxPooling1D(2, name='pool1')(x)

        # 124
        x = self._res_unit(x, 128, stage=1, block=1, stride=2)
        x = self._res_unit(x, 128, stage=1, block=2)

        # 64
        x = self._res_unit(x, 256, stage=2, block=1, stride=2)

        # 32
        x = self._res_unit(x, 256, stage=3, block=1, stride=2)

        # 32
        x = self._res_unit(x, 512, stage=4, block=1, stride=2)

        x = kl.GlobalAveragePooling1D()(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)


class ResAtrous01(DnaModel):
    """Residual network with Atrous (dilated) convolutional layers.

    Residual network with Atrous (dilated) convolutional layer in bottleneck
    units. Atrous convolutional layers allow to increase the receptive field and
    hence better model long-range dependencies.

    .. code::

        Parameters: 2,000,000
        Specification: conv[128@11]_mp[2]_resa[3x128|3x256|3x512|1x1024]_gap_do

    He et al., 'Identity Mappings in Deep Residual Networks.'
    Yu and Koltun, 'Multi-Scale Context Aggregation by Dilated Convolutions.'
    """

    def _res_unit(self, inputs, nb_filter, size=3, stride=1, atrous=1,
                  stage=1, block=1):

        name = '%02d-%02d/' % (stage, block)
        id_name = '%sid_' % (name)
        res_name = '%sres_' % (name)

        # Residual branch

        # 1x1 down-sample conv
        x = kl.BatchNormalization(name=res_name + 'bn1')(inputs)
        x = kl.Activation('relu', name=res_name + 'act1')(x)
        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(nb_filter[0], 1,
                      name=res_name + 'conv1',
                      subsample_length=stride,
                      init=self.init,
                      W_regularizer=w_reg)(x)

        # LxL conv
        x = kl.BatchNormalization(name=res_name + 'bn2')(x)
        x = kl.Activation('relu', name=res_name + 'act2')(x)
        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.AtrousConv1D(nb_filter[1], size,
                            atrous_rate=atrous,
                            name=res_name + 'conv2',
                            border_mode='same',
                            init=self.init,
                            W_regularizer=w_reg)(x)

        # 1x1 up-sample conv
        x = kl.BatchNormalization(name=res_name + 'bn3')(x)
        x = kl.Activation('relu', name=res_name + 'act3')(x)
        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(nb_filter[2], 1,
                      name=res_name + 'conv3',
                      init=self.init,
                      W_regularizer=w_reg)(x)

        # Identity branch
        if nb_filter[-1] != inputs._keras_shape[-1] or stride > 1:
            w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
            identity = kl.Conv1D(nb_filter[2], 1,
                                 name=id_name + 'conv1',
                                 subsample_length=stride,
                                 init=self.init,
                                 W_regularizer=w_reg)(inputs)
        else:
            identity = inputs

        x = kl.merge([identity, x], name=name + 'merge', mode='sum')

        return x

    def __call__(self, inputs):
        x = inputs[0]

        w_reg = kr.WeightRegularizer(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Conv1D(128, 11,
                      name='conv1',
                      init=self.init,
                      W_regularizer=w_reg)(x)
        x = kl.Activation('relu', name='act1')(x)
        x = kl.MaxPooling1D(2, name='pool1')(x)

        # 124
        x = self._res_unit(x, [32, 32, 128], stage=1, block=1, stride=2)
        x = self._res_unit(x, [32, 32, 128], atrous=2, stage=1, block=2)
        x = self._res_unit(x, [32, 32, 128], atrous=4, stage=1, block=3)

        # 64
        x = self._res_unit(x, [64, 64, 256], stage=2, block=1, stride=2)
        x = self._res_unit(x, [64, 64, 256], atrous=2, stage=2, block=2)
        x = self._res_unit(x, [64, 64, 256], atrous=4, stage=2, block=3)

        # 32
        x = self._res_unit(x, [128, 128, 512], stage=3, block=1, stride=2)
        x = self._res_unit(x, [128, 128, 512], atrous=2, stage=3, block=2)
        x = self._res_unit(x, [128, 128, 512], atrous=4, stage=3, block=3)

        # 16
        x = self._res_unit(x, [256, 256, 1024], stage=4, block=1, stride=2)

        x = kl.GlobalAveragePooling1D()(x)
        x = kl.Dropout(self.dropout)(x)

        return self._build(inputs, x)


def list_models():
    """Return the name of models in the module."""

    models = dict()
    for name, value in globals().items():
        if inspect.isclass(value) and name.lower().find('model') == -1:
            models[name] = value
    return models


def get(name):
    """Return object from module by its name."""
    return get_from_module(name, globals())
