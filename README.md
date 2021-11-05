# Dependency-Parsing-with-Neural-Networks
Training a feed-forward neural network to predict the transitions of an arc-standard dependency parser

## Goal
The input to this network will be a representation of the current state (including words on the stack and buffer). The output will be a transition (shift, left_arc, right_arc), together with a dependency relation label.
