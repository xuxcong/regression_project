Fri 9 Jun 2018
==============
- Author(s): Zheng Le Wen
- Working Directory: *Research*

The Base Model Class
--------------------
Today we wrote a base class ``ModelBase`` to ease subsequent model testing procedure. You can check it in the file *Research/model_test.py*.

``ModelBase`` is basically a wrapper for all future classification models. It will load data for you, call your model's algorithms, and collect results.
It perform the whole cross validation procedure for you. All you need to do to write your own model, is to:

- Derive it
- Override ``setup`` method. Define and initialize your model's parameters here.
- Override ``train`` method. This is where you will put the core learning algorithm of your model. Basically it is updating your params defined in ``setup``. All needed knowledge is features and labels, and these are already given as method parameters.
- Override ``predict`` method. Use your trained model to predict and return labels here, with only features given.

A framework is as follow:

.. code-block:: python

    from model_test import ModelBase
    import pandas as pd

    class SimpleModel(ModelBase):
        def setup(self):
            # Define and initialize your model's parameters here
            pass

        def train(self,features,labels):
            # Specify how to train your model here
            # All data you need will be features and labels as given in parameters
            pass

        def predict(self,features):
            # Compute and return your prediction here from given features
            # All data you need is features
            # Also model parameters should be already updated from train method

            # Initialize labels
            n = len(features)
            # Must return pandas Series
            labels = pd.Series([0]*n)
            return labels

For a running simple example, you can check the ``SimpleModel`` class in *Research/model_test.py*.

Then in main function, we can perform our experiment easily as follow:

.. code-block:: python

    from model_test import summary
    # Create model
    model = SimpleModel()
    # Run model
    results = model.run()
    # Display results
    summary(results)

Here ``summary`` method is used to format output the cross validation results.

You can simply run the script *Research/model_test.py* in a terminal to see all things running. Just run:

.. code-block:: shell

    python model_test.py

under *Research/* directory.

