using PyPlot

function generateData()
    Xtrain = Matrix(0,2);
    ytrain = [];
    var = 3
    mn = 5
    for i = linspace(-1, 1, 30);
        for j = linspace(-1, 1, 30);
            if (i-j < -0.5);
                ytrain = [ytrain; -1];
                Xtrain = [Xtrain; 50+mn+i+(var*rand(1,1)-1) mn+j+(var*rand(1,1)-1)];
            elseif (i-j > 1.5);
                ytrain = [ytrain; 1];
                Xtrain = [Xtrain; 50+mn+i+(var*rand(1,1)-1) mn+j+(var*rand(1,1)-1)];
            end
        end
    end
    return (Xtrain, ytrain);
end

function plotGraph(Xtrain, ytrain, x1, x2, name)
    figure();
    xlim(minimum(Xtrain[:,1]),maximum(Xtrain[:,1]));
    ylim(minimum(Xtrain[:,2]),maximum(Xtrain[:,2]));
    title(name);
    plot(Xtrain[ytrain.==1,1], Xtrain[ytrain.==1,2], "b+");
    plot(Xtrain[ytrain.==-1,1], Xtrain[ytrain.==-1,2], "ro");
    plot(x1, x2, "");
end
