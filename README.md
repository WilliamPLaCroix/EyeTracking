# EyeTracking Seminar Project
Final project for Saarland University Winter Semester 2023-2024 Eye Tracking seminar - [view the final report here](https://github.com/WilliamPLaCroix/EyeTracking/blob/main/Project_Report.pdf)

This project seeks to extend the research performed in Ryzhova, Škrjanec, et al. 2023 by training neural a classifier for unknown word detection.

Using a standard suite of neural network libraries, a Pytorch implementation of a MLP for binary classification was developed that exceeded the performance of the logistic regression classifier from the original paper. The original intent was to use LSTM/CNN recurrent networks for this classification task, due to their ability to capture patterns across sequential input, but performance of the comparatively simpler feedforward model ended up superior to the recurrent models. Future work to increase the robustness of recurrent models for classification may be done at a later date.

All required data and scripts for running the project are included in the repository. Data is read in from CSV, and either the notebook can be used for interactive prototyping, or the model script MLP.py can be run without arguments to perform full training runs on the various classification tasks.

## installation:

To install the required dependencies, run the following command:

```sh
pip install -r requirements.txt
```

Author: William LaCroix

Special acknowledgement goes to Margarita Ryzhova and Iza Škrjanec for their original work, as well as help in working with the dataset and developing predictive models.

License: MIT License
