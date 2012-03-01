.. _FlightTypes:

============
Flight Types
============

The following flight types may be derived by the 'FDR Flight Type' attribute.

 * COMPLETE - A complete flight, including takeoff and landing.
 * TRAINING - A complete flight where events are deliberately triggered in order to train in how to handle them. Specified by 'AFR Type' in the achieved flight record.
 * LINE_TRAINING - A training captain is on a standard commercial flight monitoring the crew. Specified by 'AFR Type' in the achieved flight record.
 * POSITIONING - Moving of the aircraft from one airport to another. Specified by 'AFR Type' in the achieved flight record.
 * FERRY - Moving of the aircraft from one airport to another. Specified by 'AFR Type' in the achieved flight record.
 * TEST - A flight where the aircraft is tested after a change is made or maintenance is performed. Specified by 'AFR Type' in the achieved flight record.
 * REJECTED_TAKEOFF - Flight with origin, but no destination as the takeoff was rejected before leaving the ground.
 * LIFTOFF_ONLY - The flight contains a liftoff, but no touchdown. The end of the flight data is missing.
 * TOUCHDOWN_ONLY - The flight contains a touchdown, but no liftoff. The beginning of the flight data is missing.
 * TOUCHDOWN_BEFORE_LIFTOFF - A touchdown has been detected before liftoff, therefore the data may contain incomplete flights.
 * ENGINE_RUN_UP - Engines are turned on, and then off again with no flight.
 * GROUND_RUN - The aircraft remained on the ground.