This GitHub repository houses the development for the final assignment of the 4th year module "Connectionist Computing" with the module code COMP30230. It focuses on the fine-tuning of a multi-layer perception model specifically designed to address the following set of questions:

1. Train an MLP with 2 inputs, 3-4+ hidden units and one output on the following examples (XOR function):
   ((0, 0), 0)
   ((0, 1), 1)
   ((1, 0), 1)
   ((1, 1), 0)
2. At the end of training, check if the MLP predicts correctly all the examples.
3. Generate 500 vectors containing 4 components each. The value of each component should be a random number between -1 and 1. These will be your input vectors. The corresponding output for each vector should be the sin() of a combination of the components. Specifically, for inputs: [x1 x2 x3 x4]
the (single component) output should be:
sin(x1-x2+x3-x4)
Now train an MLP with 4 inputs, at least 5 hidden units and one output on 400 of these examples and keep the remaining 100 for testing.
4. What is the error on training at the end? How does it compare with the error on the test set? Do you think you have learned satisfactorily?
