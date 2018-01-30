include("utils.jl")
include("models.jl")

(Xtrain, ytrain) = generateData();
x1 = collect(minimum(Xtrain[:,1]):0.01:maximum(Xtrain[:,1]));
# SMO
c = 5;
(a, w, b) = SMO(Xtrain, ytrain, c);
x2 = -(w[1]/w[2])*x1 - b/w[2];
plotGraph(Xtrain, ytrain, x1, x2, "SVM with SMO")
# Primal w/ Reguarlized Bias
w2 = reg(Xtrain, ytrain);
x22 = -(w2[1]/w2[2])*x1 - w2[3]/w2[2];
plotGraph(Xtrain, ytrain, x1, x22, "SVM with Regularized Bias")
# Primal w/ Unregularized Bias
(w3, bb) = unreg(Xtrain, ytrain);
x23 = -(w3[1]/w3[2])*x1 - getvalue(bb)/w3[2];
plotGraph(Xtrain, ytrain, x1, x23, "SVM with Unregularized Bias")
