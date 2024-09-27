% Function which performs a gradient descent compressed sensing algorithm
% on the image data.
function reconstructedImage = compressedSensing(Image, p, h, lambda, Num_iterations, ...
                                            wavelet_name, level, PLOT)
    
    % INPUTS:
    %   p       - Probability of undersampling
    %   h       - Step size
    %   lambda  - Step size
    %   PLOT    - Specify to plot graph
    
    % Create the subsampled matrix to subsample each F.T. image:
    M = rand(size(Image)) > p;
    
    % Create the y vector
    y = Image.*M;
    
    % Perform wavelet decomposition for the given wavelet and level:
    [C,S] = wavedec2(Image, level, wavelet_name);
    
    % Create lambda vector from the approximation and detail coefficients:
    Lambda = zeros(size(C));
    Lambda(prod(S(1,:))+1:end) = lambda;
    
    % Initialize the z vector
    z = ones(1,length(C));
    
    % Save some z values so that I can plot objective function at the end
    % of the process:
    z_obj_function = zeros(Num_iterations/10, length(C));
    
    % Iterate through the gradient descent algorithm via the equation from
    % class:
    for k = 1:Num_iterations
        vec1 = wavedec2(M .* waverec2(z, S, wavelet_name), level, wavelet_name);
        vec2 = wavedec2(y, level, wavelet_name);
        z = softThreshold(z-h*(vec1 - vec2), h * Lambda);
        
        if mod(k,10) == 0
            z_obj_function(k/10,:) = z;
        end
    end
    
    % Create the reconstructed image from the z
    reconstructedImage = waverec2(z, S, wavelet_name);
    
    % Create plots for the images for the report:
    if PLOT
        figure;
        subplot(2, 2, 1);
        imagesc(Image);
        title("Original Image");
        axis("square");
        colormap("gray");
        
        subplot(2, 2, 2);
        imagesc(reconstructedImage);
        title("Reconstructed Image");
        axis("square");
        colormap("gray")
        
        subplot(2, 2, 3);
        imagesc(y);
        title("Subsampled Image");
        colormap("gray");
        axis("square");
        
        % Create the objective function from the values of z at integer
        % multiples of 10 that we initialized earlier:
        objectiveFunction = zeros(1, size(z_obj_function, 1));
        for c = 1:size(z_obj_function, 1)
            objectiveFunction(c) = 0.5 * norm(M .* waverec2(z_obj_function(c, :), S, "coif3")-y, 2)^2 ... 
                                    + norm(Lambda .* z_obj_function(c, :), 1);
        end
        
        % Append the objective function plot to the other image plots:
        subplot(2, 2, 4);
        semilogy(10 : 10 : Num_iterations, objectiveFunction);
        title("Objective Function");
        xlabel("Number of Iterations");
        sgtitle("Compressed sensing with probailty = " + p + " | step size = " ...
                + h + " | \lambda = " + lambda + " | iterations = " + Num_iterations ...
                + newline + "Assume sparsity in the " + level + "-level " ...
                + WaveletName(wavelet_name) + " wavelet domain");

    end

end

