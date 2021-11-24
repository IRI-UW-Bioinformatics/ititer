ititer
======

Serial dilution assays are widespread in biology.
Typically, measurements are taken on serially diluted samples, yielding data that fit sigmoid curves.
The position of the sigmoid curve is captured by its inflection point, or **inflection titer**, which can then be compared among samples.

.. image:: sigmoid.png
    :align: center
    :width: 60 %

Alternatively, the dilution at which the measurement drops below a cut-off value can be measured.
This is known as an **endpoint titer**.

Investigators often measure all samples at many dilutions (typically 8-12).
This is usually overkill as the underlying sigmoid curve can be recovered by many fewer dilutions.

**ititer** uses Bayesian hierarchical modelling to pool inference of sigmoid curve characteristics that are shared among samples, which further reduces the number dilutions required to infer inflection or endpoint titers.

Applied to a SARS-CoV-2 and human seasonal coronavirus ELISA dataset, just 3 dilutions yielded the same inflection and endpoint titers as 8 dilutions (`Pattinson et al. 2021 <https://www.medrxiv.org/content/10.1101/2021.09.13.21263523v1>`_).

Highlights
----------

* Simple interface aimed at enabling investigators with basic knowledge of python to conduct analyses.
* Substantially decrease workload associated with serial dilution assays.
* Flexibly incorporate different prior knowledge about experimental systems to facilitate inference.

Contents
--------

.. toctree::
   :maxdepth: 2

   quick
   tutorial
   inference
   commands

* :ref:`genindex`

Installation
------------

ititer is available from pip:

.. code-block:: bash

    $ pip install ititer

Citation
--------

If you use this software please cite:

  `A method to reduce ELISA serial dilution assay workload applied to SARS-CoV-2 and seasonal HCoVs <https://www.medrxiv.org/content/10.1101/2021.09.13.21263523v1>`_

  David Pattinson, Peter Jester, Lizheng Guan, Seiya Yamayoshi, Shiho Chiba, Robert Presler Jr., Hongyu Rao, Kiyoko Iwatsuki-Horimoto, Nobuhiro Ikeda, Masao Hagihara, Tomoyuki Uchida, Keiko Mitamura, Peter Halfmann, Gabriele Neumann, Yoshihiro Kawaoka

  *medRxiv* 2021.09.13.21263523; doi: https://doi.org/10.1101/2021.09.13.21263523
