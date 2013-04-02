====================================
Derived Parameter Naming Convention
====================================

---------------------
Method
---------------------

Each Parameter, including derived parameters will have the following form:

Noun Identifier Qualifier

Where::

* Noun is the name of the aircraft system or flight measurement. In simple terms it can be said to be the quantity that can be measured in a defined unit.
* Identifier defines one of a set of multiple systems and are always listed in parentheses. They can range from Left and Right sensors denoted by (L) and (R) while Captain and First Officer
defined as (Capt) and (FO)
* Qualifier is needed when the naming is to be made more precise. They can include words like "Max" and and even verbs like "Selected". All qualifier verbs shall be in past tense. Hence, 
"Selected" is allowed but "Select" or "Selection" are not. 

---------------------
Examples
---------------------

**Roll Rate**

The parameter having the Unit in this case is Roll which is recorded in degrees. We are interested to derive the rate of change of roll, which in this case shall give us a derived unit of degrees/second. Hence, following the naming convention will result in the following:

Noun: Roll (in degrees)
Identifier: None in this case
Qualifier: Rate (in seconds)

Parameter Name: RollRate

**Airspeed Minus V2 For 3Sec**

This parameter computes the airspeed on takeoff relative (subtraction) to V2 over a 3 second window. Hence, the parameter naming convention gives us the following name:

Noun: Airspeed (kts)
Identifier: None
Qualifier: MinusV2For3sec (in order to make the naming more precise)

Parameter Name: AirspeedMinusV2For3Sec

---------------------
References
---------------------
* `Polaris Parameter and Key Point Value Naming Convention <http://www.flightdatacommunity.com/polaris-suite/>`_. 

