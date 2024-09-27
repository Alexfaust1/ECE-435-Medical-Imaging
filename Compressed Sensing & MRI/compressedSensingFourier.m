function [objF, z] = compressedSensingFourier(Image, p, h, lambda, iterations, wavelet_name, ...
                                    level, PLOT)
    % Keep same inputs as compressed sensing function when signal prior is
    % assumed sparse in image domain...
    
    % Create the subsampled matrix as in other domain:
    M = rand(size(Image)) > p;

    % Create the y vector:
    y = Image.*M

    % Decompose wavelet by the corresponding level:
    [C, S] = wavedec2(Image, level, wavelet_name);

    % Create lambda from approximation and detail coefficients:
    Lambda = zeros(size(C));
    Lambda(prod(S(1, :)) + 1 : end) = lambda;

    % Initialize z vector:
    z = ones(1, length(C));

    % Choose to save some z values to plot objective function at end:
    z_obj_function = zeros(iterations/10, length(C));

    % Iterate through gradient descent with the NEW descent iteration:
    for k = 1:iterations
        % Build up the gradient term slowly in terms:
        % "y" terms:
        FFTinv_y = abs(ifft2(y));
        W_FFTinv_y = wavedec2(FFTinv_y, level, wavelet_name);

        % "z" terms:
        Winv_z = waverec2(z, S, wavelet_name);
        FT_Winv_z = abs(fft2(Winv_z));
        M_FT_Winv_z = M * FT_Winv_z;
        FTinv_M_FT_Winv_z = abs(ifft(M_FT_Winv_z));
        W_FTinv_M_FT_Winv_z = wavedec2(FTinv_M_FT_Winv_z, level, wavelet_name);

        z = softThreshold(z - h*(W_FTinv_M_FT_Winv_z - W_FFTinv_y), h*Lambda);
        
        % Track the z's for objective function:
        if mod(k, 10) == 0
            z_obj_function(k/10, :) = z;
        end

    end

    % Take the final z and reconstruct the image from this z:
    reconstructedImage = waverec2(z, S, wavelet_name);

    if PLOT
        figure;
        subplot(2, 2, 1);
        imagesc(Image);
        title("Original Image");
        axis("square");
        colormap("gray");
        
        subplot(2, 2, 2);
        imagesc(abs(reconstructedImage));
        title("Reconstructed Image");
        axis("square");
        colormap("gray")
        
        subplot(2, 2, 3);
        imagesc(abs(y));
        title("Subsampled Image");
        colormap("gray");
        axis("square");
        
        % Create objective function from values of z:
        objF = zeros(1, size(z_obj_function, 1));
        for c = 1:size(z_obj_function, 1)
            objF(c) = 0.5 * norm(M .* waverec2(z_obj_function(c, :), S, "coif3")-y, 2)^2 ...
                                    + norm(lambda .* z_obj_function(c, :), 1);
        end

        % Add objective function to other images:
        subplot(2, 2, 4);
        semilogy(10 : 10 : iterations, objF);
        title("Objective Function");
        xlabel("Number of Iterations");
        sgtitle("Compressed sensing with probailty = " + p + " | step size = " ...
                + h + " | \lambda = " + lambda + " | iterations = " + iterations ...
                + newline + "Assume sparsity in the " + level + "-level " ...
                + WaveletName(wavelet_name) + " wavelet domain");
        
    end

end