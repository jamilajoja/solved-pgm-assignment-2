Download Link: https://assignmentchef.com/product/solved-pgm-assignment-2
<br>
<h1>Pictorial Structures</h1>

The goal of this exercise is to implement a simplified version of the pictorial structures model, although using an efficient algorithm, and test it on a pedestrian dataset. You are going to use a star model over (upper and lower) legs, head, and torso as follows:

Each <em>L<sub>i </sub></em>in this distribution is actually a two-dimensional random variable representing image coordinates of the center of each body part: <em>L<sub>i </sub></em>= (<em>x<sub>i</sub>,y<sub>i</sub></em>). Scale and rotation is not considered here. Unary factors represent likelihoods <em>p<sub>i</sub></em>(<em>L<sub>i</sub></em>) and pairwise factors stand for kinematic priors <em>p<sub>ij</sub></em>(<em>L<sub>i</sub>,L<sub>j</sub></em>). You will learn the priors from a training dataset. Then, you will perform inference on test images, for which you are given the corresponding likelihood maps.

The first step is to download <strong>assignment02-data.zip </strong>from the lecture’s website and import the <strong>data.mat</strong>. The variable likelihoods{k,i} is the 2D likelihood map <em>p<sub>i</sub></em>(<em>L<sub>i</sub></em>) for test image number <em>k</em>. The images can be found in the directory testset for reference. The index <em>i </em>is defined as follows:

<em>i           </em>Body part

1    Lower Left leg 2 Upper Left leg

<ul>

 <li>Upper Right leg</li>

 <li>Lower Right leg</li>

 <li>Head</li>

 <li>Torso</li>

</ul>

The variable train contains the training data. More precisely, each row <em>k </em>of train{i} represents the coordinates <em>L<sup>k</sup><sub>i </sub></em>of body part <em>i </em>in training image <em>k</em>.

<h1>1         Learning Kinematic Priors</h1>

We set the prior as a 2D Gaussian <em>p<sub>ij</sub></em>(<em>L<sub>i</sub>,L<sub>j</sub></em>) ∼ N(<em>L<sub>i </sub></em>− <em>T<sub>ij</sub></em>(<em>L<sub>j</sub></em>);0<em>,</em>Σ<em><sub>ij</sub></em>) and we define the transformation <em>T<sub>ij</sub></em>(<em>L<sub>j</sub></em>) = <em>L<sub>j </sub></em>+ <em>µ<sub>ij</sub></em>. This is a reasonable approximation for small joint rotations, as it is the case for pedestrians. The parameters to be determined for each prior are thus the covariance matrix Σ<em><sub>ij </sub></em>and the mean <em>µ<sub>ij</sub></em>.

Implement the function pairwisePots = learnPairwisePots(train) which, for each body part <em>i</em>, computes the maximum-likelihood estimate of <em>µ<sub>ij </sub></em>as a row vector pairwisePots{i,1} and of Σ<em><sub>ij </sub></em>as pairwisePots{i,2}. Note that there are only potentials between the torso and the remaining body parts, hence we assume <em>j </em>= 6 is fixed. Use the built-in Matlab functions mean and cov.

<h1>2         Maximal Marginal States</h1>

The goal is to compute maximal marginals of the model using the sumproduct algorithm. To this end, implement the corresponding function maxstates

= sumproduct(pairwisePots, unaryPots). It returns a 6×2 matrix of <em>x,y </em>coordinates.

<ul>

 <li>Note that the graph is not a chain but a tree, so you have to think about the correct message scheduling.</li>

 <li>You will need computations like <em>f</em>(<em>L<sub>i</sub></em>) = <sup>P</sup><em><sub>L</sub></em><em>j </em>N(<em>L<sub>i </sub></em>− <em>T<sub>ij</sub></em>(<em>L<sub>j</sub></em>))<em>g</em>(<em>L<sub>j</sub></em>) in your code. However, summing over <em>L<sub>j </sub></em>is impractical due to the large state space. That is why you should implement the sum as a convolution (N ∗ (<em>g </em>◦ <em>T<sub>ij</sub></em><sup>−1</sup>))(<em>L<sub>i</sub></em>) := <sup>P</sup><em><sub>s </sub></em>N(<em>L<sub>i </sub></em>− <em>s</em>)<em>g</em>(<em>T<sub>ij</sub></em><sup>−1</sup>(<em>s</em>)) with <em>s </em>= <em>T<sub>ij</sub></em>(<em>L<sub>j</sub></em>) using built-in functions. We additionally assume a diagonal covariance so that the Gaussian is separable (simply ignore the off-diagonal entries of Σ<em><sub>ij</sub></em>). The Matlab functions fspecial, conv2 and the prepared file <strong>m </strong>(it shifts a given image for a given 2D offset) will be useful here. Take care when computing messages in the opposite direction.</li>

 <li>You can visualize your maxima using <strong>m</strong>. You can compare your results for the first 10 images with the official solution in the directory solution.</li>

 <li>Hint: Recall that we work with (<em>x,y</em>) vectors but Matlab indexes its matrices by (row, column).</li>

</ul>

<h1>3         Modes</h1>

The goal is to compute the maximum state of the joint distribution using the min-sum algorithm (i.e. using negated log potentials). Implement the function maxstates = minsum(pairwisePots, unaryPots). It returns a 6×2 matrix of <em>x,y </em>coordinates.

<ul>

 <li>For reasons of efficiency, we compute the minima using the generalized distance transform</li>

</ul>

DT−log[<em>g</em>(<em>T</em><em>ij</em>−1(·))](<em>L</em><em>i</em>) = min<em>s δ</em>(<em>L</em><em>i,s</em>) − log[<em>g</em>(<em>T</em><em>ij</em>−1(<em>s</em>))]

= min−log[N(<em>L<sub>i </sub></em>− <em>T<sub>ij</sub></em>(<em>L<sub>j</sub></em>))] − log[<em>g</em>(<em>L<sub>j</sub></em>)]<em>.</em>

<em>L<sub>j</sub></em>

You can find the code in <strong>DT.m</strong>, the function takes the covariance matrix of the Gaussian as the second argument.

<ul>

 <li>Unfortunately, DT does not give you the argmin, only the min. For this reason, you cannot backtrack and you need to implement the min-sum algorithm as in the lecture on max-sum algorithm earlier, i.e. computing all messages (two messages per edge) and taking a node-wise minimum. This will probably fail on potential ties (multiple modes) but that is fine in this exercise.</li>

</ul>

<h1>4         Evaluation</h1>

Now you are going to evaluate the model as a person detector by writing a script <strong>evaluation.m</strong>. To keep it simple, fix a bounding box of size 80x200px around the torso (horizontally centered at it, vertically offset in 1:2 ratio). A predicted bounding box is considered <em>correct </em>if it overlaps more than 50% with a ground-truth bounding box, otherwise the bounding box is considered a false positive detection. Ground truth can be found in the variable GT in the supplied mat file, each row is a rectangle [<em>x</em>1<em>,y</em>1<em>,w,h</em>]. Bounding box overlap is computed using the <strong>boxoverlap.m </strong>function which is provided for your convenience.

Compute bounding boxes for each test image using three ways of choosing the torso:

<ul>

 <li>Torso as the result of min-sum.</li>

 <li>Torso as the result of sum-product.</li>

 <li>Torso as the maximum of a torso’s likelihood, which corresponds to using no model at all.</li>

</ul>

Your script should output the accuracy of each method, i.e. the percentage of correctly predicted bounding boxes. Note that you can visualize your bounding boxes and detections using <strong>drawmaxima.m</strong>.