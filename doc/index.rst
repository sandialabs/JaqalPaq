.. toctree::
   :maxdepth: 2
   :caption: Contents:

   jaqalpaq
   glossary

Python Jaqal Programming Package (JaqalPaq)
===========================================

API Reference
-------------
`JaqalPaq <https://gitlab.com/jaqal/jaqalpaq>`_ currently consists of the
:mod:`jaqalpaq` Python package, and its subpackages.

* The :mod:`jaqalpaq.core` package implements an object representation of scheduled
  quantum circuits. It supports programmatically constructing and manipulating circuits.
* The :mod:`jaqalpaq.parser` package parses Jaqal source files into
  :class:`jaqalpaq.core.Circuit` objects.
* The :mod:`jaqalpaq.generator` package generates Jaqal code that implements the
  quantum circuit described by a :class:`jaqalpaq.core.Circuit` object.
* The :mod:`jaqalpaq.emulator` package provides noiseless emulation of Jaqal code.

The following subpackages are not part of the basic language features, and must be
installed separately from the  `JaqalPaq-extras repository
<https://gitlab.com/jaqal/jaqalpaq-extras>`_.

* The :mod:`jaqalpaq.scheduler` package modifies circuits to execute more gates in
  parallel, without changing the function of the circuit or breaking the restrictions
  of the QSCOUT hardware.
* The :mod:`jaqalpaq.transpilers.cirq`, :mod:`jaqalpaq.transpilers.projectq`,
  :mod:`jaqalpaq.transpilers.qiskit`, :mod:`jaqalpaq.transpilers.quil`, and
  :mod:`jaqalpaq.transpilers.tket` packages allow conversion between :mod:`jaqalpaq.core`
  objects and their counterparts in other popular quantum software development frameworks.

Additionally, the top-level :mod:`jaqalpaq` package provides a few useful imports that
don't fit within the scope of any of the above subpackages: the :exc:`jaqalpaq.error.JaqalError`
class and a collection of :data:`RESERVED_WORDS`.


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
* :ref:`glossary`
