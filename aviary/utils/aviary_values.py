'''
Define utilities for using aviary values with associated units and testing
for compatibility with aviary metadata dictionary.

Utilities
---------
Units : type alias
    define a type hint for associated units

ValueAndUnits : type alias
    define a type hint for a single value paired with its associated units

OptionalValueAndUnits : type alias
    define a type hint for an optional single value paired with its associated units

class AviaryValues
    define a collection of named values with associated units
'''
import operator
from enum import EnumMeta
from functools import reduce

import numpy as np
from openmdao.utils.units import convert_units as _convert_units

from aviary.utils.named_values import NamedValues, get_items, get_keys, get_values
from aviary.variable_info.variable_meta_data import _MetaData
from aviary.utils.utils import isiterable


class AviaryValues(NamedValues):
    '''
    Define a collection of aviary values with associated units and aviary tests.
    '''

    def set_val(self, key, val, units='unitless', meta_data=_MetaData):
        '''
        Update the named value and its associated units.

        Note, specifying units of `None` or units of any type other than `str` will raise
        `Typerror`.

        Parameters
        ----------
        key : str
            the name of the item

        val : Any
            the new value of the item

        units : str ('unitless')
            the units associated with the new value, if any

        Raises
        ------
        TypeError
            if units of `None` were specified or units of any type other than `str`
        '''

        my_val = val
        if key in meta_data.keys():
            expected_types = meta_data[key]['types']
            if not isinstance(expected_types, tuple):
                expected_types = (expected_types, )

            # If provided val is not in expected types, see if it can be casted to one of
            # them (e.g. cast int to float).
            if not isinstance(val, expected_types):
                # Prefer casting to Enum if possible
                # Special handling to access an Enum member from either the member name
                # or its value.
                if EnumMeta in expected_types:
                    if isiterable(val):
                        my_val = [self._convert_to_enum(
                            item, _type) for item in val]
                    else:
                        my_val = self._convert_to_enum(val, _type)
                else:
                    for _type in expected_types:
                        try:
                            if isiterable(val):
                                if isinstance(val, np.ndarray):
                                    my_val = np.array([_type(item) for item in val])
                                else:
                                    my_val = type(val)([_type(item) for item in val])
                            else:
                                my_val = _type(val)
                        except (ValueError, TypeError):
                            # try another value in expected_types
                            pass
                        else:
                            break

            # Special handling if the variable is supposed to be an array
            # If the item is supposed to be an iterable...
            # if meta_data[key]['multivalue']:
            #     # but the provided value is not...
            #     if not isiterable(my_val):
            #         # make object the correct iterable
            #         if tuple in expected_types:
            #             my_val = (my_val,)
            #         else:
            #             dtype = type(meta_data[key]['default_value'])
            #             my_val = np.array([my_val], dtype)

            self._check_type(key, my_val, meta_data=meta_data)
            self._check_units_compatibility(key, my_val, units, meta_data=meta_data)

        super().set_val(key=key, val=my_val, units=units)

    def _check_type(self, key, val, meta_data=_MetaData):
        """
        Check that provided val is the correct type. If val is iterable, also check each
        individual index
        """

        expected_types = meta_data[key]['types']
        if expected_types is None:
            # MetaData item has no type requirement.
            return

        # If data is iterable, check that it is allowed to be.
        # Variables flagged multivalue can be lists or numpy arrays even if not specified
        if isiterable(val):
            types = expected_types
            if meta_data[key]['multivalue']:
                if isinstance(expected_types, tuple):
                    types = (list, np.ndarray, *expected_types)
                else:
                    types = (list, np.ndarray, expected_types)
            if not isinstance(val, types):
                raise TypeError(
                    f'{key} is of type(s) {types} but you have provided a value of type '
                    f'{type(val)}.')

        # numpy arrays have special typings. Convert to list using standard Python types
        # numpy arrays do not allow mixed types, only have to check one entry
        # empty arrays do not need this check
        if isinstance(val, np.ndarray) and len(val) > 0:
            val = val.tolist()
            while isiterable(val):
                val = val[0]

        # if val is not iterable, make it a list (checks assume val is iterable)
        if not isiterable(val):
            val = [val]
        # if val is an iterable, flatten it so we can easily loop over every entry
        else:
            val = _flatten_iters(val)

        for item in val:
            has_bool = False  # needs some fancy shenanigans because bools will register as ints
            if (isinstance(expected_types, type)):
                if expected_types is bool:
                    has_bool = True
            elif bool in expected_types:
                has_bool = True
            if (not isinstance(item, expected_types)) or (
                    (has_bool == False) and (isinstance(item, bool))):
                raise TypeError(
                    f'{key} is of type(s) {meta_data[key]["types"]} but you '
                    f'have provided a value of type {type(item)}.')

    def _check_units_compatibility(self, key, val, units, meta_data=_MetaData):
        expected_units = meta_data[key]['units']

        try:
            # NOTE the value here is unimportant, we only care if OpenMDAO will
            # convert the units
            _convert_units(10, expected_units, units)
        except ValueError:
            raise ValueError(
                f'The units {units} which you have provided for {key} are invalid.')
        except TypeError:
            raise TypeError(
                f'The base units of {key} are {expected_units}, and you have tried to set {key} with units of {units}, which are not compatible.')
        except BaseException:
            raise KeyError('There is an unknown error with your units.')

    def _convert_to_enum(self, val, enum_type):
        if isinstance(val, str):
            try:
                # see if str maps to ENUM value
                return enum_type(val)
            except ValueError:
                # str instead maps to ENUM name
                return enum_type[val.upper()]


def _flatten_iters(iterable):
    """Flattens iterables of any type and size"""
    for item in iterable:
        try:
            yield from iter(item)
        except TypeError:
            yield item
