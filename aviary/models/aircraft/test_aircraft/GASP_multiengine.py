"""
Fake aircraft model used for testing GASP multiengine capability only. Will be replaced by
hybrid-electric regional turboprop when ready. Based on large_single_aisle_1_GASP.
"""

from aviary.subsystems.propulsion.engine_deck import EngineDeck
from aviary.utils.preprocessors import preprocess_propulsion
from aviary.utils.process_input_decks import create_vehicle
from aviary.variable_info.variables import Aircraft

inputs, _ = create_vehicle('large_single_aisle_1_GASP')
inputs.set_val(Aircraft.Engine.SCALE_FACTOR, 0.4)
_engine1 = EngineDeck('engine1', inputs)
inputs.set_val(Aircraft.Engine.SCALE_FACTOR, 0.6)
_engine2 = EngineDeck('engine2', inputs)

engines = [_engine1, _engine2]
preprocess_propulsion(inputs, engines)
