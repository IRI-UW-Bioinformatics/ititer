ititer
======

A python module for analyzing data derived from serial dilution assays.

Serial dilution assays are widespread in biology.
Typically, measurements are taken on serially diluted samples, yielding data that fit sigmoid curves.
The position of the sigmoid curve is captured by its inflection point, or **inflection titer**, which are then compared among samples.

.. image:: sigmoid.png
    :align: center
    :width: 50 %

Alternatively, the dilution at which the measurement drops below a cutoff value can be measured.
This is known as an **endpoint titer**.

Investigators often measure all samples at many dilutions (typically 8-12).
This is usually overkill as the underlying sigmoid curve can be recovered by many fewer dilutions.

**ititer** uses Bayesian hierarchical modelling to pool inference of sigmoid curve characteristics that are shared among samples, drastically reducing the number dilutions required to infer inflection or endpoint titers.

When applied to a SARS-CoV-2 and human seasonal coronavirus dataset, just 3 dilutions yielded the same inflection and endpoint titers as 8 dilutions. <Cite XXX>.

Highlights
----------

* Simple interface aimed at enabling investiagtors with basic knowledge of python to conduct analyses.
* Substantially decrease workload associated with serial dilution assays.
* Flexibly incorporate different prior knowledge about experimental systems to faciliate inference.

Contents
--------

.. toctree::
   :maxdepth: 2

   tutorial
   commands

Installation
------------

ititer is available from pip:

.. code-block:: bash

    $ pip install ititer


Citation
--------

If you use this software please cite: XXX


Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`
