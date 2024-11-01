
The goal of the project was to predict the success of high school students. Data was obtained from [this](https://archive.ics.uci.edu/dataset/320/student+performance) source. There are a total of 1,044 examples and 30 predictors. Out of these 1,044 examples, 649 are grades from Portuguese, and 395 are grades from Mathematics. Some of the predictors are binary, some are multi-class, some are nominal, and some are continuous. Grades range from 0 to 20.

The output variable is the number of points achieved in the third year of high school. For a more detailed analysis, the output variable was observed in three ways:

* As a continuous value
* As a binary value: passed (points >= 10) or failed (points < 10)
* As a grade: A (16-20), B (14-15), C (12-13), D (10-11), F (0-9)
  
Some models are better at solving classification problems, while others are more suited for regression. Also, sometimes it might be more relevant to perform a simplified prediction of whether a student passed or failed, rather than predicting their exact grade. Grades are categorized as F (failed) and A, B, C, D, which provide a more detailed analysis of the scores of students who passed. Depending on the purpose of the examination, different interpretations of success may be useful.
