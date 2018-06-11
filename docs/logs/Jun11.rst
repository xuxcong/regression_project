Mon 11 Jun 2018
===============
We want to incorporate cabin information into our model. But simply linearly separating the number in cabin code like "A12" into ranges would work well probabily, since "A12" and "B12" may not align vertically. Besides, all cabins on A deck are located in one side of the ship, while cabins of other deck distributed across the ship.

So maybe we will need to hard coded the location information into the model.

I am gonna test against SVC, NuSVC and LinearSVC, BernoulliNB, GaussianNB models.

We are studying this `kernel <https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy/notebook>`_ on kaggle which achieved over 82% on test set. It do not consider `Cabin` variable in its model because there are too many null values (over 77%). So even after we incorporate the `Cabin` information, at most only about 22% of the predictions will be improved. But we could try incorporating it, maybe.

The data preprocessing, exploration, and visualization and other many steps of the above kernel may be transferred to our report.



