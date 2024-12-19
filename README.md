# allen-cahn-pinns
A try of solving Allen-Cahn equation using Physical-Informed Neural Network, using periodical boundary condition. I trying for both 1d and 2d condition, both get the accuracy of a relative error under 5%. 
The requirements are provided in requirements.txt, 1d training of allen cahn equation is just simple, and I apply the calculation just on my latptop with apple silicon m1 mac air with full cache of 8GB, it takes about 5-10 minutes.
The 2d pinns solver of allen cahn using the warm up techniques and do pre training, with which to initialize the more complex and instable equation, which make it more likely to converge to correct solution that I get using numerical method instead of trival solution.
But the 2d pinns training need about 40w data interier points, to make it much faster, I use a 2060 GPU as training device the accerate rate is about 20, The whole training process then can be done in about 20 minutes.
